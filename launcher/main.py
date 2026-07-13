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

from .env_builder import (
    bootstrap_launcher_env,
    detect_app_root,
)
from .event_log import EventLog
from .pid_store import launcher_activity_log_path
from .process_manager import start_api, start_dashboard, stop_service
from .scroll_frame import ScrollableFrame
from .status_snapshot import collect_snapshot
from .ui_asr_dialog import open_asr_manager_dialog
from .ui_settings_dialog import open_settings_dialog
from .ui_status_panel import StatusPanel

_AUTO_REFRESH_MS = 2000
_POLL_LOG_MS = 100


def _app_root() -> Path:
    return detect_app_root()


def _format_service_error(name: str, message: str) -> str:
    """Make common dependency failures actionable for the user."""
    lower = message.lower()
    hints: list[str] = []
    if "node" in lower or "npm" in lower:
        hints.append("Installera Node.js LTS (inkl. npm) från https://nodejs.org")
        hints.append("Kör sedan: cd webui && npm install")
    elif "node_modules" in lower or "package.json" in lower or "webui" in lower:
        hints.append("Kontrollera att mappen webui/ finns i app-roten")
        hints.append("Kör: cd webui && npm install")
    elif "uvicorn" in lower or "fastapi" in lower:
        hints.append("Kör «Installera / Reparera allt» eller: pip install -e \".[api]\"")
    if not hints:
        return message
    return message + "\n\nNästa steg:\n• " + "\n• ".join(hints)


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
        self._busy_buttons: list[ttk.Button] = []

        self._scroll = ScrollableFrame(self)
        self._scroll.pack(fill=tk.BOTH, expand=True)

        self.status_panel = StatusPanel(
            self._scroll.inner,
            self.event_log,
            activity_log_path=log_path,
        )
        self.status_panel.pack(fill=tk.X)

        self._build_buttons(self._scroll.inner)
        self.event_log.phase("launcher", "Launcher startad")
        self.status_panel.activity.load_all()
        self._refresh_status()
        self._schedule_poll_log()
        self._schedule_auto_refresh()

    def _add_section(
        self,
        parent: tk.Misc,
        title: str,
        specs: list[tuple[str, object, bool]],
    ) -> None:
        frame = ttk.LabelFrame(parent, text=title, padding=8)
        frame.pack(fill=tk.X, padx=12, pady=(8, 4))
        for label, cmd, busy_sensitive in specs:
            btn = ttk.Button(frame, text=label, command=cmd)
            btn.pack(fill=tk.X, pady=2)
            if busy_sensitive:
                self._busy_buttons.append(btn)

    def _build_buttons(self, parent: tk.Misc) -> None:
        self._add_section(
            parent,
            "Tjänster",
            [
                ("Starta API", lambda: self._run_service_action("api", "start"), True),
                ("Stoppa API", lambda: self._run_service_action("api", "stop"), True),
                (
                    "Starta Dashboard",
                    lambda: self._run_service_action("dashboard", "start"),
                    True,
                ),
                (
                    "Stoppa Dashboard",
                    lambda: self._run_service_action("dashboard", "stop"),
                    True,
                ),
            ],
        )
        self._add_section(
            parent,
            "Verktyg",
            [
                ("Inställningar…", self._open_settings, False),
                ("Doctor / hälsokontroll", self._run_doctor, True),
                ("Hantera ASR / transkribering", self._open_asr_manager, True),
                ("Öppna CLI (PowerShell)", self._open_cli, False),
                ("Öppna outputs-mapp", self._open_outputs, False),
                ("Öppna logs-mapp", self._open_logs, False),
            ],
        )
        self._add_section(
            parent,
            "Underhåll",
            [
                ("Installera / reparera allt", self._provision, True),
            ],
        )

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        for btn in self._busy_buttons:
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
                shown = _format_service_error(name, msg)
                title = "API" if name == "api" else "Dashboard"
                self.after(0, lambda m=shown, t=title: messagebox.showerror(t, m))
            finally:
                self.after(0, self._on_action_done)

        self._set_busy(True)
        action_sv = "Startar" if action == "start" else "Stoppar"
        self.event_log.phase(f"{name}.{action}", f"{action_sv} {name}")
        threading.Thread(target=work, daemon=True).start()

    def _on_action_done(self) -> None:
        self._set_busy(False)
        self._refresh_status()

    def _run_doctor(self) -> None:
        if self._busy:
            return
        self._set_busy(True)
        self.event_log.phase("doctor", "Kör hälsokontroller")

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

    def _open_settings(self) -> None:
        self.cfg = load_user_config(_app_root())
        open_settings_dialog(
            self,
            _app_root(),
            self.event_log,
            on_saved=self._refresh_status,
            on_provision=self._provision,
            on_open_asr=self._open_asr_manager,
        )
        self.event_log.info("Öppnade inställningar", phase="launcher")

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
        self.event_log.info("Öppnade ASR-hanterare", phase="launcher")

    def _open_cli(self) -> None:
        from .cli import open_cli_cmd

        open_cli_cmd()
        self.event_log.info("Öppnade PowerShell CLI", phase="launcher")

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
            "Installera / reparera",
            (
                f"Detta laddar ner och installerar allt som behövs för profil '{profile}':\n\n"
                "• Python virtual environment (.venv)\n"
                "• Pip-paket (API m.m.)\n"
                "• faster-whisper, whisperx och transkriberingsmodeller\n"
                "• ffmpeg (om det saknas)\n"
                "• Node.js-beroenden för webui (npm install)\n"
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
                            "Installera / reparera",
                            "Alla komponenter installerades.",
                        )
                    else:
                        messagebox.showwarning(
                            "Installera / reparera",
                            "Vissa steg misslyckades. Se aktivitetsloggen.",
                        )

                self.after(0, done)
            except Exception as exc:
                msg = str(exc)
                self.event_log.error(msg, phase="provision")

                def failed() -> None:
                    self._set_busy(False)
                    messagebox.showerror(
                        "Installera / reparera",
                        _format_service_error("provision", msg),
                    )

                self.after(0, failed)

        self._set_busy(True)
        self.event_log.phase("provision", f"Installerar profil '{profile}'")
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
