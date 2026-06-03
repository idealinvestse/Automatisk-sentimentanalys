"""Tkinter launcher hub for Windows (no extra GUI dependencies)."""

from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from src.install.preflight import run_preflight
from src.install.user_config import default_user_config_path, load_user_config

from .env_builder import build_child_env, resolve_python, working_directory
from .process_manager import (
    service_status,
    start_api,
    start_dashboard,
    stop_service,
)


def _app_root() -> Path:
    return Path(os.environ.get("SENTIMENT_APP_ROOT", Path.cwd())).resolve()


class LauncherApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Sentimentanalys")
        self.geometry("420x480")
        self.cfg = load_user_config(_app_root())
        self._build_ui()
        self._refresh_status()

    def _build_ui(self) -> None:
        pad = {"padx": 12, "pady": 6}
        ttk.Label(self, text="Automatisk sentimentanalys", font=("Segoe UI", 14, "bold")).pack(
            **pad
        )
        self.status_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.status_var, justify=tk.LEFT).pack(fill=tk.X, **pad)

        frame = ttk.Frame(self)
        frame.pack(fill=tk.BOTH, expand=True, **pad)

        buttons = [
            ("Start API", self._start_api),
            ("Stop API", lambda: self._stop("api")),
            ("Start Dashboard", self._start_dashboard),
            ("Stop Dashboard", lambda: self._stop("dashboard")),
            ("Configure (Setup Hub)", self._open_setup_hub),
            ("Doctor / Health check", self._run_doctor),
            ("Open CLI (PowerShell)", self._open_cli),
            ("Open outputs folder", self._open_outputs),
            ("Open logs folder", self._open_logs),
            ("Repair dependencies", self._repair),
        ]
        for label, cmd in buttons:
            ttk.Button(frame, text=label, command=cmd).pack(fill=tk.X, pady=3)

        ttk.Label(
            self,
            text=f"Config: {default_user_config_path(self.cfg.portable_mode, self.cfg.resolved_app_root())}",
            wraplength=380,
            font=("Segoe UI", 8),
        ).pack(side=tk.BOTTOM, pady=8)

    def _refresh_status(self) -> None:
        self.cfg = load_user_config(_app_root())
        api = service_status(self.cfg, "api")
        dash = service_status(self.cfg, "dashboard")
        self.status_var.set(
            f"API ({self.cfg.services.api_port}): {api}\n"
            f"Dashboard ({self.cfg.services.dashboard_port}): {dash}\n"
            f"Profile: {self.cfg.sentiment_profile} | Device: {self.cfg.device}"
        )

    def _start_api(self) -> None:
        try:
            info = start_api(self.cfg)
            messagebox.showinfo("API", f"Started (pid {info.pid})")
        except Exception as e:
            messagebox.showerror("API", str(e))
        self._refresh_status()

    def _start_dashboard(self) -> None:
        try:
            info = start_dashboard(self.cfg)
            messagebox.showinfo(
                "Dashboard",
                f"Started (pid {info.pid})\nhttp://localhost:{self.cfg.services.dashboard_port}",
            )
        except Exception as e:
            messagebox.showerror("Dashboard", str(e))
        self._refresh_status()

    def _stop(self, name: str) -> None:
        stop_service(self.cfg, name)
        self._refresh_status()

    def _run_doctor(self) -> None:
        report = run_preflight(self.cfg)
        lines = [f"{'OK' if c.ok else 'FAIL'}: {c.name} — {c.message}" for c in report.checks]
        messagebox.showinfo("Doctor", "\n".join(lines[:20]))

    def _open_setup_hub(self) -> None:
        self.cfg = load_user_config(_app_root())
        root = working_directory(self.cfg)
        py = resolve_python(self.cfg)
        subprocess.Popen(
            [str(py), "-m", "streamlit", "run", "app/setup_hub.py"],
            cwd=str(root),
            env=build_child_env(self.cfg),
        )

    def _open_cli(self) -> None:
        from .cli import open_cli_cmd

        open_cli_cmd()

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

    def _repair(self) -> None:
        if not messagebox.askyesno("Repair", "Re-install pip packages for current profile?"):
            return
        from src.install.config_schema import InstallProfile

        from .cli import repair_cmd

        try:
            repair_cmd(profile=InstallProfile(self.cfg.install_profile.value))
            messagebox.showinfo("Repair", "Done")
        except Exception as e:
            messagebox.showerror("Repair", str(e))


def main() -> None:
    os.environ.setdefault("SENTIMENT_APP_ROOT", str(Path.cwd().resolve()))
    app = LauncherApp()
    app.mainloop()


if __name__ == "__main__":
    main()
