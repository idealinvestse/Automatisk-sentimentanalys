"""Tkinter status panel widgets for the launcher."""

from __future__ import annotations

import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import scrolledtext, ttk
from typing import TYPE_CHECKING

from .event_log import EventLog, LogEvent
from .status_snapshot import LauncherSnapshot, ServiceSnapshot, ServiceState, SystemSnapshot

if TYPE_CHECKING:
    from collections.abc import Callable


_STATE_STYLES: dict[ServiceState, str] = {
    ServiceState.RUNNING: "Status.Running.TLabel",
    ServiceState.LISTENING: "Status.Listening.TLabel",
    ServiceState.STARTING: "Status.Starting.TLabel",
    ServiceState.DEGRADED: "Status.Warning.TLabel",
    ServiceState.ERROR: "Status.Error.TLabel",
    ServiceState.STOPPED: "Status.Stopped.TLabel",
}

_STATUS_COLORS: dict[str, str] = {
    "Status.Running.TLabel": "#0d7a3e",
    "Status.Listening.TLabel": "#0b6e8c",
    "Status.Starting.TLabel": "#9a6b00",
    "Status.Warning.TLabel": "#b45309",
    "Status.Error.TLabel": "#b91c1c",
    "Status.Stopped.TLabel": "#6b7280",
}


def configure_status_styles(root: tk.Misc) -> None:
    """Register ttk label styles (Windows requires TLabel-based style names)."""
    style = ttk.Style(root)
    for name, color in _STATUS_COLORS.items():
        style.configure(name, foreground=color)
    style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"))
    style.configure("CardTitle.TLabel", font=("Segoe UI", 10, "bold"))
    style.configure("Meta.TLabel", font=("Segoe UI", 8), foreground="#4b5563")
    style.configure("Url.TLabel", foreground="#2563eb")
    style.configure("UrlIdle.TLabel", foreground="#111827")


class ServiceCard(ttk.LabelFrame):
    def __init__(self, master: tk.Misc, title: str) -> None:
        super().__init__(master, text=title, padding=8)
        self._url: str | None = None
        self.state_lbl = ttk.Label(self, text="—", style="Status.Stopped.TLabel")
        self.state_lbl.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 6))

        fields = [
            ("PID", "pid"),
            ("URL", "url"),
            ("Port", "port"),
            ("Process", "process"),
            ("Health", "health"),
        ]
        self._values: dict[str, ttk.Label] = {}
        for i, (label, key) in enumerate(fields):
            ttk.Label(self, text=f"{label}:", style="Meta.TLabel").grid(
                row=1 + i, column=0, sticky=tk.W, padx=(0, 8)
            )
            val = ttk.Label(self, text="—")
            val.grid(row=1 + i, column=1, sticky=tk.W)
            self._values[key] = val
            if key == "url":
                val.bind("<Button-1>", self._open_url)

    def _open_url(self, _event: tk.Event[object]) -> None:
        if self._url:
            webbrowser.open(self._url)

    def update_from(self, snap: ServiceSnapshot) -> None:
        style = _STATE_STYLES.get(snap.state, "Status.Stopped.TLabel")
        self.state_lbl.configure(text=snap.state_label, style=style)
        self._url = snap.url if snap.port_open else None
        self._values["pid"].configure(text=str(snap.pid) if snap.pid else "—")
        url_text = snap.url if snap.port_open else "—"
        self._values["url"].configure(
            text=url_text,
            style="Url.TLabel" if snap.port_open else "UrlIdle.TLabel",
            cursor="hand2" if snap.port_open else "",
        )
        self._values["port"].configure(
            text=f"{snap.host}:{snap.port} ({'öppen' if snap.port_open else 'stängd'})"
        )
        self._values["process"].configure(text="levande" if snap.process_alive else "död/saknas")
        if snap.health_ok is None:
            health = "n/a"
        else:
            health = "ok" if snap.health_ok else "fel"
        self._values["health"].configure(text=health)


class SystemPanel(ttk.LabelFrame):
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, text="System", padding=8)
        self._labels: dict[str, ttk.Label] = {}
        rows = [
            ("Launcher", "launcher_root"),
            ("App root", "app_root"),
            ("Config", "config_path"),
            ("Python", "python_exe"),
            ("Profil", "profile"),
            ("Enhet", "device"),
            ("LLM", "llm"),
            ("Nycklar", "secrets"),
        ]
        for i, (title, key) in enumerate(rows):
            ttk.Label(self, text=f"{title}:", style="Meta.TLabel").grid(
                row=i, column=0, sticky=tk.NW, padx=(0, 8), pady=1
            )
            val = ttk.Label(self, text="—", wraplength=340)
            val.grid(row=i, column=1, sticky=tk.W, pady=1)
            self._labels[key] = val

    def update_from(self, sys: SystemSnapshot) -> None:
        self._labels["launcher_root"].configure(text=str(sys.launcher_root))
        self._labels["app_root"].configure(text=str(sys.app_root))
        self._labels["config_path"].configure(text=str(sys.config_path))
        venv = "venv OK" if sys.venv_ok else "ingen venv"
        self._labels["python_exe"].configure(text=f"{sys.python_exe} ({venv})")
        self._labels["profile"].configure(
            text=f"{sys.install_profile} / sentiment: {sys.sentiment_profile}"
        )
        self._labels["device"].configure(text=sys.device)
        self._labels["llm"].configure(
            text=f"{'på' if sys.llm_enabled else 'av'} (API v{sys.api_version})"
        )
        or_s = "OR ✓" if sys.openrouter_configured else "OR ✗"
        hf_s = "HF ✓" if sys.huggingface_configured else "HF ✗"
        self._labels["secrets"].configure(text=f"{or_s}  {hf_s}")


class ActivityLogPanel(ttk.LabelFrame):
    def __init__(
        self,
        master: tk.Misc,
        event_log: EventLog,
        *,
        on_clear: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(master, text="Aktivitet", padding=6)
        self._log = event_log
        toolbar = ttk.Frame(self)
        toolbar.pack(fill=tk.X, pady=(0, 4))
        clear_btn = ttk.Button(toolbar, text="Rensa visning", command=self._clear)
        clear_btn.pack(side=tk.RIGHT)
        clear_btn.bind(
            "<Enter>",
            lambda _e: clear_btn.configure(
                cursor="question_arrow",
            ),
        )
        self.text = scrolledtext.ScrolledText(
            self,
            height=8,
            font=("Consolas", 9),
            state=tk.DISABLED,
            wrap=tk.WORD,
        )
        self.text.pack(fill=tk.X)
        self._on_clear = on_clear

    def _clear(self) -> None:
        self._log.clear()
        self.text.configure(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        self.text.configure(state=tk.DISABLED)
        if self._on_clear:
            self._on_clear()

    def append_events(self, events: list[LogEvent]) -> None:
        if not events:
            return
        self.text.configure(state=tk.NORMAL)
        for ev in events:
            self.text.insert(tk.END, ev.format_line() + "\n")
        self.text.see(tk.END)
        self.text.configure(state=tk.DISABLED)

    def load_all(self) -> None:
        self.text.configure(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        for ev in self._log.entries():
            self.text.insert(tk.END, ev.format_line() + "\n")
        self.text.see(tk.END)
        self.text.configure(state=tk.DISABLED)


class StatusPanel(ttk.Frame):
    """Top section: service cards, system info, activity log, footer."""

    def __init__(
        self,
        master: tk.Misc,
        event_log: EventLog,
        *,
        activity_log_path: Path | None = None,
    ) -> None:
        configure_status_styles(master)
        super().__init__(master)
        self._activity_log_path = activity_log_path

        header = ttk.Frame(self)
        header.pack(fill=tk.X, padx=12, pady=(10, 4))
        ttk.Label(header, text="Automatisk sentimentanalys", style="Header.TLabel").pack(
            side=tk.LEFT
        )

        cards = ttk.Frame(self)
        cards.pack(fill=tk.X, padx=12, pady=4)
        self.api_card = ServiceCard(cards, "API")
        self.api_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        self.dash_card = ServiceCard(cards, "Dashboard")
        self.dash_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))

        self.system_panel = SystemPanel(self)
        self.system_panel.pack(fill=tk.X, padx=12, pady=4)

        self.activity = ActivityLogPanel(self, event_log)
        self.activity.pack(fill=tk.X, padx=12, pady=4)

        self.footer_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.footer_var, style="Meta.TLabel").pack(
            fill=tk.X, padx=12, pady=(0, 6)
        )

    def apply_snapshot(self, snap: LauncherSnapshot) -> None:
        self.api_card.update_from(snap.api)
        self.dash_card.update_from(snap.dashboard)
        self.system_panel.update_from(snap.system)
        log_hint = ""
        if self._activity_log_path is not None:
            log_hint = f"  ·  logg: {self._activity_log_path.name}"
        self.footer_var.set(
            f"Senast uppdaterad: {snap.collected_at}  ·  auto-uppdatering var 2 s{log_hint}"
        )
