"""Tkinter dialog for managing ASR packages and model downloads."""

from __future__ import annotations

import threading
import tkinter as tk
from collections.abc import Callable
from tkinter import messagebox, ttk

from src.install.config_schema import UserConfig

from .asr_manager import asr_status_for_config, format_asr_report_lines, run_asr_setup
from .event_log import EventLog


class AsrManagerDialog(tk.Toplevel):
    """Modal dialog: install faster-whisper/whisperx and prefetch models."""

    def __init__(
        self,
        master: tk.Misc,
        cfg: UserConfig,
        event_log: EventLog,
        *,
        on_complete: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(master)
        self.title("ASR / Transkribering")
        self.geometry("460x420")
        self.resizable(False, False)
        self.transient(master)
        self.grab_set()

        self.cfg = cfg
        self.event_log = event_log
        self._on_complete = on_complete
        self._busy = False

        pad = {"padx": 12, "pady": 4}
        frame = ttk.Frame(self, padding=8)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            frame,
            text="Hantera faster-whisper, whisperx och transkriberingsmodeller",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor=tk.W, **pad)

        self.status_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.status_var, wraplength=400).pack(anchor=tk.W, **pad)

        self.model_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.model_var, foreground="#4b5563").pack(anchor=tk.W, **pad)

        backends_frame = ttk.LabelFrame(frame, text="Backends att förladda", padding=8)
        backends_frame.pack(fill=tk.X, **pad)

        self._backend_vars: dict[str, tk.BooleanVar] = {}
        for key, label in (
            ("faster", "faster-whisper (standard)"),
            ("whisperx", "whisperx (alignment + diarization)"),
            ("transformers", "transformers (HF pipeline)"),
        ):
            var = tk.BooleanVar(value=key in ("faster", "whisperx"))
            self._backend_vars[key] = var
            ttk.Checkbutton(backends_frame, text=label, variable=var).pack(anchor=tk.W)

        self.progress_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.progress_var, foreground="#6b7280").pack(
            anchor=tk.W, **pad
        )

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, **pad)

        self._buttons: list[ttk.Button] = []
        for label, cmd in (
            ("Installera paket", lambda: self._start_action(packages=True, models=False)),
            ("Ladda ner modeller", lambda: self._start_action(packages=False, models=True)),
            ("Allt (paket + modeller)", lambda: self._start_action(packages=True, models=True)),
        ):
            btn = ttk.Button(btn_frame, text=label, command=cmd)
            btn.pack(fill=tk.X, pady=2)
            self._buttons.append(btn)

        ttk.Button(btn_frame, text="Stäng", command=self._close).pack(fill=tk.X, pady=(8, 0))

        self._refresh_status()
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _selected_backends(self) -> list[str]:
        return [k for k, var in self._backend_vars.items() if var.get()]

    def _refresh_status(self) -> None:
        status = asr_status_for_config(self.cfg)
        self.status_var.set(status.summary())
        self.model_var.set(
            f"Modell: {status.model_name} (revision: {self.cfg.asr.revision}) · "
            f"Cache: {status.hf_cache_dir}"
        )

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        for btn in self._buttons:
            btn.configure(state=state)

    def _start_action(self, *, packages: bool, models: bool) -> None:
        if self._busy:
            return
        backends = self._selected_backends()
        if models and not backends:
            messagebox.showwarning("ASR", "Välj minst en backend för modellnedladdning.")
            return

        action = []
        if packages:
            action.append("paket")
        if models:
            action.append("modeller")
        if not messagebox.askyesno(
            "ASR",
            f"Detta kommer att hämta {' och '.join(action)}.\n\nKräver internet. Fortsätt?",
        ):
            return

        self._set_busy(True)
        phase = "asr.setup"
        self.event_log.phase(phase, f"ASR: {' + '.join(action)}")
        self.progress_var.set("Startar…")

        def work() -> None:
            try:
                report = run_asr_setup(
                    self.cfg,
                    backends=backends,
                    install_packages=packages,
                    download_models=models,
                    progress=lambda msg: self.event_log.info(msg, phase=phase),
                )
                for ok, line in format_asr_report_lines(report):
                    if ok:
                        self.event_log.info(line, phase=phase)
                    else:
                        self.event_log.error(line, phase=phase)

                def done() -> None:
                    self._set_busy(False)
                    self._refresh_status()
                    self.progress_var.set("Klart." if report.ok else "Vissa steg misslyckades.")
                    if report.ok:
                        messagebox.showinfo("ASR", "ASR-uppsättning slutförd.")
                    else:
                        messagebox.showwarning(
                            "ASR", "Vissa steg misslyckades. Se aktivitetsloggen."
                        )
                    if self._on_complete:
                        self._on_complete()

                self.after(0, done)
            except Exception as exc:
                msg = str(exc)
                self.event_log.error(msg, phase=phase)

                def failed() -> None:
                    self._set_busy(False)
                    self.progress_var.set("Fel.")
                    messagebox.showerror("ASR", msg)

                self.after(0, failed)

        threading.Thread(target=work, daemon=True).start()

    def _close(self) -> None:
        if self._busy and not messagebox.askyesno("ASR", "Nedladdning pågår. Stäng ändå?"):
            return
        self.grab_release()
        self.destroy()


def open_asr_manager_dialog(
    master: tk.Misc,
    cfg: UserConfig,
    event_log: EventLog,
    *,
    on_complete: Callable[[], None] | None = None,
) -> None:
    """Open the ASR management dialog."""
    AsrManagerDialog(master, cfg, event_log, on_complete=on_complete)
