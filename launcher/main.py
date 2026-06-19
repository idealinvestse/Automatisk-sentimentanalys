"""Tkinter launcher hub for Windows (no extra GUI dependencies)."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from src.install.preflight import run_preflight
from src.install.user_config import load_user_config

from .env_builder import bootstrap_launcher_env, build_child_env, detect_app_root, resolve_python, working_directory
from .event_log import EventLog
from .pid_store import launcher_activity_log_path
from .process_manager import start_api, start_dashboard, stop_service
from .scroll_frame import ScrollableFrame
from .status_snapshot import collect_snapshot
from .ui_asr_dialog import open_asr_manager_dialog
from .ui_status_panel import StatusPanel

_AUTO_REFRESH_MS = 2000
_POLL_LOG_MS = 100


def _app_root() -> Path:
    return detect_app_root()


class LauncherApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Sentimentanalys — Kontrollpanel")
        self.geometry("540x720")
        self.minsize(480, 400)
        self.cfg = load_user_config(_app_root())
        log_path = launcher_activity_log_path(self.cfg)
        self.event_log = EventLog(log_path=log_path)
        self._busy = False
        self._action_buttons: list[ttk.Button] = []

        self._scroll = ScrollableFrame(self)
        self._scroll.pack(fill=tk.BOTH, expand=True)

        self.status_panel = StatusPanel(
            self._scroll.inner,
            self.event_log,
            activity_log_path=log_path,
        )
        self.status_panel.pack(fill=tk.X)

        self._build_buttons(self._scroll.inner)
        self.event_log.phase("launcher", "Launcher started")
        self.status_panel.activity.load_all()
        self._refresh_status()
        self._schedule_poll_log()
        self._schedule_auto_refresh()

    def _build_buttons(self, parent: tk.Misc) -> None:
        pad = {"padx": 12, "pady": 4}
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, **pad)

        specs: list[tuple[str, object]] = [
            ("Start API", lambda: self._run_service_action("api", "start")),
            ("Stop API", lambda: self._run_service_action("api", "stop")),
            ("Start Dashboard", lambda: self._run_service_action("dashboard", "start")),
            ("Stop Dashboard", lambda: self._run_service_action("dashboard", "stop")),
            ("Configure (Setup Hub)", self._open_setup_hub),
            ("Doctor / Health check", self._run_doctor),
            ("Hantera ASR / Transkribering", self._open_asr_manager),
            ("Open CLI (PowerShell)", self._open_cli),
            ("Open outputs folder", self._open_outputs),
            ("Open logs folder", self._open_logs),
            ("Installera / Reparera allt", self._provision),
        ]
        for label, cmd in specs:
            btn = ttk.Button(frame, text=label, command=cmd)
            btn.pack(fill=tk.X, pady=2)
            if "Start" in label or "Stop" in label:
                self._action_buttons.append(btn)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        for btn in self._action_buttons:
            btn.configure(state=state)

    def _schedule_poll_log(self) -> None:
        self._poll_log()
        self.after(_POLL_LOG_MS, self._schedule_poll_log)

    def _poll_log(self) -> None:
        events = self.event_log.poll_queue()
        self.status_panel.activity.append_events(events)

    def _schedule_auto_refresh(self) -> None:
        if not self._busy:
            self._refresh_status()
        self.after(_AUTO_REFRESH_MS, self._schedule_auto_refresh)

    def _refresh_status(self) -> None:
        self.cfg = load_user_config(_app_root())
        snap = collect_snapshot(self.cfg, launcher_root=_app_root())
        self.status_panel.apply_snapshot(snap)

    def _run_service_action(self, name: str, action: str) -> None:
        if self._busy:
            return

        def work() -> None:
            try:
                if action == "start":
                    if name == "api":
                        start_api(self.cfg, log=self.event_log)
                    else:
                        start_dashboard(self.cfg, log=self.event_log)
                else:
                    stop_service(self.cfg, name, log=self.event_log)
            except Exception as exc:
                msg = str(exc)
                self.event_log.error(msg, phase=f"{name}.{action}")
                self.after(0, lambda m=msg, n=name: messagebox.showerror(n.upper(), m))
            finally:
                self.after(0, self._on_action_done)

        self._set_busy(True)
        self.event_log.phase(f"{name}.{action}", f"{action.capitalize()} {name}")
        threading.Thread(target=work, daemon=True).start()

    def _on_action_done(self) -> None:
        self._set_busy(False)
        self._refresh_status()

    def _run_doctor(self) -> None:
        if self._busy:
            return
        self._set_busy(True)
        self.event_log.phase("doctor", "Running health checks")

        def work() -> None:
            report = run_preflight(self.cfg)
            for c in report.checks:
                msg = f"{c.name}: {c.message}"
                if c.detail:
                    msg += f" ({c.detail})"
                if c.ok:
                    self.event_log.info(msg, phase="doctor")
                else:
                    self.event_log.error(msg, phase="doctor")

            def done() -> None:
                self._set_busy(False)
                if not report.ok:
                    messagebox.showwarning(
                        "Doctor",
                        "Vissa kontroller misslyckades. Se aktivitetsloggen.",
                    )

            self.after(0, done)

        threading.Thread(target=work, daemon=True).start()

    def _open_setup_hub(self) -> None:
        self.cfg = load_user_config(_app_root())
        root = working_directory(self.cfg)
        py = resolve_python(self.cfg)
        subprocess.Popen(
            [str(py), "-m", "streamlit", "run", "app/setup_hub.py"],
            cwd=str(root),
            env=build_child_env(self.cfg),
        )
        self.event_log.info("Opened Setup Hub (Streamlit)", phase="launcher")

    def _open_asr_manager(self) -> None:
        if self._busy:
            return
        self.cfg = load_user_config(_app_root())
        open_asr_manager_dialog(
            self,
            self.cfg,
            self.event_log,
            on_complete=self._refresh_status,
        )
        self.event_log.info("Opened ASR manager", phase="launcher")

    def _open_cli(self) -> None:
        from .cli import open_cli_cmd

        open_cli_cmd()
        self.event_log.info("Opened PowerShell CLI", phase="launcher")

    def _open_folder(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", str(path)], check=False)

    def _open_outputs(self) -> None:
        self._open_folder(self.cfg.resolved_app_root() / self.cfg.paths.outputs)

    def _open_logs(self) -> None:
        self._open_folder(self.cfg.resolved_logs_dir())

    def _provision(self) -> None:
        if self._busy:
            return

        profile = self.cfg.install_profile.value
        if not messagebox.askyesno(
            "Installera / Reparera",
            (
                f"Detta laddar ner och installerar allt som behövs för profil '{profile}':\n\n"
                "• Python virtual environment (.venv)\n"
                "• Pip-paket (API, dashboard m.m.)\n"
                "• faster-whisper, whisperx och transkriberingsmodeller\n"
                "• ffmpeg (om det saknas)\n"
                "• user_config.yaml (om den saknas)\n\n"
                "Kräver internetanslutning. Fortsätt?"
            ),
        ):
            return

        from src.install.config_schema import InstallProfile
        from src.install.provision import run_provision

        def work() -> None:
            try:
                report = run_provision(
                    self.cfg,
                    InstallProfile(profile),
                    progress=lambda msg: self.event_log.info(msg, phase="provision"),
                )
                for step in report.steps:
                    msg = f"{step.name}: {step.message}"
                    if step.detail:
                        msg += f" ({step.detail})"
                    if step.ok:
                        self.event_log.info(msg, phase="provision")
                    else:
                        self.event_log.error(msg, phase="provision")

                def done() -> None:
                    self._set_busy(False)
                    self._refresh_status()
                    if report.ok:
                        messagebox.showinfo(
                            "Installera / Reparera",
                            "Alla komponenter installerades.",
                        )
                    else:
                        messagebox.showwarning(
                            "Installera / Reparera",
                            "Vissa steg misslyckades. Se aktivitetsloggen.",
                        )

                self.after(0, done)
            except Exception as exc:
                msg = str(exc)
                self.event_log.error(msg, phase="provision")

                def failed() -> None:
                    self._set_busy(False)
                    messagebox.showerror("Installera / Reparera", msg)

                self.after(0, failed)

        self._set_busy(True)
        self.event_log.phase("provision", f"Installing profile '{profile}'")
        threading.Thread(target=work, daemon=True).start()


def main() -> None:
    root = bootstrap_launcher_env()
    try:
        app = LauncherApp()
        app.mainloop()
    except Exception:
        import traceback

        (root / "launcher_crash.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()