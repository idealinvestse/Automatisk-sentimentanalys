"""Tkinter settings dialog — replaces Streamlit Setup Hub."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from src.install.config_schema import InstallProfile, UserConfig
from src.install.secrets_win import SecretKind, secret_status, user_secret_file

from .event_log import EventLog
from .scroll_frame import ScrollableFrame
from .settings_service import (
    _DEVICES,
    _LOG_LEVELS,
    _SECRET_KINDS,
    _SENTIMENT_PROFILES,
    clear_secret,
    export_bundle,
    import_bundle,
    load_settings,
    run_doctor,
    save_draft,
    save_secret_permanent,
    validate_draft,
)
from .ui_asr_dialog import open_asr_manager_dialog


class SettingsDialog(tk.Toplevel):
    """Modal settings window with tabbed UserConfig + secrets editing."""

    def __init__(
        self,
        master: tk.Misc,
        app_root: Path,
        event_log: EventLog,
        *,
        on_saved: Callable[[], None] | None = None,
        on_provision: Callable[[], None] | None = None,
        on_open_asr: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(master)
        self.title("Inställningar")
        self.geometry("620x580")
        self.minsize(520, 480)
        self.transient(master)
        self.grab_set()

        self._app_root = app_root
        self._event_log = event_log
        self._on_saved = on_saved
        self._on_provision = on_provision
        self._on_open_asr = on_open_asr

        snap = load_settings(app_root)
        self._baseline = snap.config.model_copy(deep=True)
        self._draft = snap.config.model_copy(deep=True)
        self._config_path = snap.config_path
        self._pending_secrets: dict[SecretKind, str] = {}
        self._secret_entries: dict[SecretKind, ttk.Entry] = {}
        self._dirty = False

        self._vars: dict[str, tk.Variable] = {}
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        outer = ttk.Frame(self, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)

        path_lbl = ttk.Label(outer, text=f"Config: {self._config_path}", foreground="#6b7280")
        path_lbl.pack(anchor=tk.W, pady=(0, 4))

        self._notebook = ttk.Notebook(outer)
        self._notebook.pack(fill=tk.BOTH, expand=True)

        self._build_all_tabs()
        self._populate_from_draft()

        btn_row = ttk.Frame(outer)
        btn_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(btn_row, text="Spara", command=self._save).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btn_row, text="Avbryt", command=self._on_close).pack(side=tk.RIGHT)

    def _mark_dirty(self, *_args: object) -> None:
        self._dirty = True

    def _add_trace(self, var: tk.Variable) -> None:
        var.trace_add("write", self._mark_dirty)

    def _scroll_tab(self, parent: tk.Misc) -> ttk.Frame:
        scroll = ScrollableFrame(parent)
        scroll.pack(fill=tk.BOTH, expand=True)
        return scroll.inner

    def _labeled_entry(self, parent: tk.Misc, label: str, key: str, *, row: int) -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W)
        var = tk.StringVar()
        self._vars[key] = var
        self._add_trace(var)
        entry = ttk.Entry(parent, textvariable=var, width=48)
        entry.grid(row=row, column=1, sticky=tk.EW, padx=(0, 8), pady=2)
        parent.columnconfigure(1, weight=1)
        return entry

    def _labeled_combo(
        self, parent: tk.Misc, label: str, key: str, values: tuple[str, ...] | list[str], *, row: int
    ) -> ttk.Combobox:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W)
        var = tk.StringVar()
        self._vars[key] = var
        self._add_trace(var)
        combo = ttk.Combobox(parent, textvariable=var, values=list(values), state="readonly", width=44)
        combo.grid(row=row, column=1, sticky=tk.EW, padx=(0, 8), pady=2)
        return combo

    def _labeled_check(self, parent: tk.Misc, label: str, key: str, *, row: int) -> ttk.Checkbutton:
        var = tk.BooleanVar()
        self._vars[key] = var
        self._add_trace(var)
        cb = ttk.Checkbutton(parent, text=label, variable=var)
        cb.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)
        return cb

    def _build_all_tabs(self) -> None:
        tabs: list[tuple[str, Callable[[ttk.Frame], None]]] = [
            ("Allmänt", self._build_general_tab),
            ("Sökvägar", self._build_paths_tab),
            ("Tjänster", self._build_services_tab),
            ("ASR", self._build_asr_tab),
            ("LLM", self._build_llm_tab),
            ("Nycklar", self._build_secrets_tab),
            ("API", self._build_api_tab),
            ("Alerting", self._build_alerting_tab),
            ("Avancerat", self._build_advanced_tab),
        ]
        for title, builder in tabs:
            frame = ttk.Frame(self._notebook, padding=4)
            self._notebook.add(frame, text=title)
            builder(frame)

    def _build_general_tab(self, parent: ttk.Frame) -> None:
        inner = self._scroll_tab(parent)
        row = 0
        self._labeled_combo(
            inner,
            "Installationsprofil (pip)",
            "install_profile",
            [p.value for p in InstallProfile],
            row=row,
        )
        row += 1
        self._labeled_check(inner, "Portabelt läge (config i ./user_data)", "portable_mode", row=row)
        row += 1
        self._labeled_combo(inner, "Sentimentprofil", "sentiment_profile", _SENTIMENT_PROFILES, row=row)
        row += 1
        self._labeled_combo(inner, "Enhet", "device", _DEVICES, row=row)
        row += 1
        self._labeled_combo(inner, "Loggnivå", "log_level", _LOG_LEVELS, row=row)
        row += 1
        ttk.Label(
            inner,
            text="Byte av installationsprofil kan kräva Installera/Reparera.",
            foreground="#b45309",
            wraplength=520,
        ).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=8)
        row += 1
        ttk.Button(inner, text="Kör Installera / Reparera", command=self._trigger_provision).grid(
            row=row, column=0, sticky=tk.W
        )

    def _build_paths_tab(self, parent: ttk.Frame) -> None:
        inner = self._scroll_tab(parent)
        fields = (
            ("outputs", "Outputs"),
            ("hf_cache", "HF-cache (HF_HOME)"),
            ("llm_cache", "LLM-cache"),
            ("logs", "Loggar (valfritt)"),
            ("incoming_calls", "Inkommande samtal (valfritt)"),
            ("state_dir", "State (valfritt)"),
        )
        for i, (key, label) in enumerate(fields):
            self._labeled_entry(inner, label, f"paths.{key}", row=i)
        ttk.Label(inner, text="App root", foreground="#6b7280").grid(row=len(fields), column=0, sticky=tk.W, pady=(8, 0))
        ttk.Label(inner, textvariable=tk.StringVar(value=str(self._app_root))).grid(
            row=len(fields), column=1, sticky=tk.W
        )

    def _build_services_tab(self, parent: ttk.Frame) -> None:
        inner = self._scroll_tab(parent)
        self._labeled_check(inner, "API aktiverad", "services.api_enabled", row=0)
        self._labeled_entry(inner, "API-värd", "services.api_host", row=1)
        self._labeled_entry(inner, "API-port", "services.api_port", row=2)
        self._labeled_check(inner, "Dashboard aktiverad", "services.dashboard_enabled", row=3)
        self._labeled_entry(inner, "Dashboard-port", "services.dashboard_port", row=4)
        self._labeled_combo(inner, "Dashboard-UI", "services.dashboard_ui", ("nicegui",), row=5)

    def _build_asr_tab(self, parent: ttk.Frame) -> None:
        inner = self._scroll_tab(parent)
        self._labeled_combo(inner, "Backend", "asr.backend", ("faster", "transformers", "whisperx"), row=0)
        self._labeled_entry(inner, "Modell", "asr.model", row=1)
        self._labeled_combo(
            inner, "Revision", "asr.revision", ("standard", "strict", "subtitle"), row=2
        )
        self._labeled_entry(inner, "Språk", "asr.language", row=3)
        self._labeled_entry(inner, "Hotwords-fil", "asr.hotwords_file", row=4)
        self._labeled_combo(
            inner, "Preprocess-läge", "asr.preprocess_mode", ("off", "basic", "callcenter"), row=5
        )
        ttk.Button(inner, text="Öppna ASR-hanterare…", command=self._open_asr_manager).grid(
            row=6, column=0, sticky=tk.W, pady=12
        )

    def _build_llm_tab(self, parent: ttk.Frame) -> None:
        inner = self._scroll_tab(parent)
        ttk.Label(
            inner,
            text="Externa LLM-anrop skickar transkript till tredje part. Aktivera endast med lämpligt samtycke.",
            foreground="#b45309",
            wraplength=520,
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 8))
        self._labeled_check(inner, "Aktivera LLM-analys", "llm.enabled", row=1)
        self._labeled_combo(inner, "Provider", "llm.provider", ("openrouter", "groq"), row=2)

        self._llm_openrouter_status = tk.StringVar(value="")
        ttk.Label(inner, text="OpenRouter API-nyckel").grid(row=3, column=0, sticky=tk.W)
        or_frame = ttk.Frame(inner)
        or_frame.grid(row=3, column=1, sticky=tk.EW, pady=2)
        or_entry = ttk.Entry(or_frame, width=40, show="*")
        or_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        or_entry.bind("<KeyRelease>", lambda _e: self._on_secret_typed("openrouter"))
        self._secret_entries["openrouter"] = or_entry
        ttk.Label(inner, textvariable=self._llm_openrouter_status, foreground="#6b7280").grid(
            row=4, column=1, sticky=tk.W
        )
        or_btn_row = ttk.Frame(inner)
        or_btn_row.grid(row=5, column=1, sticky=tk.W, pady=(0, 8))
        ttk.Button(
            or_btn_row,
            text="Spara OpenRouter-nyckel permanent",
            command=self._save_openrouter_key_now,
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(or_btn_row, text="Rensa", command=lambda: self._clear_secret("openrouter")).pack(
            side=tk.LEFT
        )

        self._labeled_entry(inner, "Standardmodell", "llm.default_model", row=6)
        self._labeled_entry(inner, "Budget per samtal (USD)", "llm.cost_budget_per_call", row=7)
        self._labeled_check(inner, "Anonymisera före LLM", "llm.anonymize_before_llm", row=8)
        self._labeled_check(inner, "Logga externa anrop", "llm.log_external_calls", row=9)
        self._refresh_openrouter_status()

    def _build_secrets_tab(self, parent: ttk.Frame) -> None:
        inner = self._scroll_tab(parent)
        self._secret_status_var = tk.StringVar()
        ttk.Label(inner, textvariable=self._secret_status_var, justify=tk.LEFT).grid(
            row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 4)
        )
        ttk.Label(
            inner,
            text="OpenRouter-nyckel konfigureras under fliken LLM. Här: Hugging Face och Groq.",
            foreground="#6b7280",
            wraplength=520,
        ).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(0, 8))
        labels = {
            "huggingface": "Hugging Face-token",
            "groq": "Groq API-nyckel",
        }
        row = 2
        for kind in ("huggingface", "groq"):
            ttk.Label(inner, text=labels[kind]).grid(row=row, column=0, sticky=tk.W)
            entry = ttk.Entry(inner, width=48, show="*")
            entry.grid(row=row, column=1, sticky=tk.EW, pady=2)
            entry.bind("<KeyRelease>", lambda _e, k=kind: self._on_secret_typed(k))
            self._secret_entries[kind] = entry
            btn_frame = ttk.Frame(inner)
            btn_frame.grid(row=row + 1, column=1, sticky=tk.W, pady=(0, 8))
            ttk.Button(btn_frame, text="Rensa", command=lambda k=kind: self._clear_secret(k)).pack(
                side=tk.LEFT, padx=(0, 4)
            )
            ttk.Button(
                btn_frame,
                text="Spara permanent",
                command=lambda k=kind: self._save_secret_now(k),
            ).pack(side=tk.LEFT)
            row += 2
        self._refresh_secret_status()

    def _build_api_tab(self, parent: ttk.Frame) -> None:
        inner = self._scroll_tab(parent)
        self._labeled_entry(inner, "API-nyckel (SENTIMENT_API_KEY)", "runtime.api.api_key", row=0)
        self._labeled_entry(inner, "CORS origins (kommaseparerade)", "runtime.api.cors_origins", row=1)
        self._labeled_entry(inner, "Media root (API_MEDIA_ROOT)", "runtime.api.media_root", row=2)
        self._labeled_entry(inner, "Rate limit (req/min, 0=av)", "runtime.api.rate_limit_rpm", row=3)
        self._labeled_check(inner, "Redis-cache", "runtime.api.use_redis_cache", row=4)
        self._labeled_entry(inner, "Redis URL", "runtime.api.redis_url", row=5)
        self._labeled_check(
            inner, "Tillåt klient-LLM-nyckel i request", "runtime.api.allow_client_llm_key", row=6
        )
        self._labeled_entry(
            inner, "Dashboard → API URL", "runtime.dashboard.api_base_url", row=7
        )
        self._labeled_check(inner, "Dev-läge (testlabb)", "runtime.dashboard.dev_mode", row=8)

    def _build_alerting_tab(self, parent: ttk.Frame) -> None:
        inner = self._scroll_tab(parent)
        self._labeled_check(inner, "Webhook aktiverad", "runtime.alerting.webhook_enabled", row=0)
        self._labeled_entry(inner, "Webhook URL", "runtime.alerting.webhook_url", row=1)
        self._labeled_entry(inner, "Timeout (s)", "runtime.alerting.timeout_seconds", row=2)
        self._labeled_entry(inner, "Max retries", "runtime.alerting.max_retries", row=3)
        self._labeled_entry(
            inner, "Circuit breaker-tröskel", "runtime.alerting.circuit_breaker_threshold", row=4
        )

    def _build_advanced_tab(self, parent: ttk.Frame) -> None:
        inner = self._scroll_tab(parent)
        ttk.Button(inner, text="Exportera config…", command=self._export_config).grid(row=0, column=0, sticky=tk.W, pady=4)
        ttk.Button(inner, text="Importera config…", command=self._import_config).grid(row=1, column=0, sticky=tk.W, pady=4)
        ttk.Separator(inner, orient=tk.HORIZONTAL).grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=8)
        ttk.Button(inner, text="Kör Doctor / hälsokontroll", command=self._run_doctor).grid(
            row=3, column=0, sticky=tk.W, pady=4
        )
        self._doctor_text = tk.Text(inner, height=12, width=70, state=tk.DISABLED)
        self._doctor_text.grid(row=4, column=0, columnspan=2, sticky=tk.NSEW, pady=4)

    def _get_var(self, key: str) -> tk.Variable:
        return self._vars[key]

    def _populate_from_draft(self) -> None:
        d = self._draft
        mapping: dict[str, object] = {
            "install_profile": d.install_profile.value,
            "portable_mode": d.portable_mode,
            "sentiment_profile": d.sentiment_profile,
            "device": d.device,
            "log_level": d.log_level,
            "paths.outputs": d.paths.outputs,
            "paths.hf_cache": d.paths.hf_cache,
            "paths.llm_cache": d.paths.llm_cache,
            "paths.logs": d.paths.logs,
            "paths.incoming_calls": d.paths.incoming_calls,
            "paths.state_dir": d.paths.state_dir,
            "services.api_enabled": d.services.api_enabled,
            "services.api_host": d.services.api_host,
            "services.api_port": str(d.services.api_port),
            "services.dashboard_enabled": d.services.dashboard_enabled,
            "services.dashboard_port": str(d.services.dashboard_port),
            "services.dashboard_ui": d.services.dashboard_ui,
            "asr.backend": d.asr.backend,
            "asr.model": d.asr.model,
            "asr.revision": d.asr.revision,
            "asr.language": d.asr.language,
            "asr.hotwords_file": d.asr.hotwords_file,
            "asr.preprocess_mode": d.asr.preprocess_mode,
            "llm.enabled": d.llm.enabled,
            "llm.provider": d.llm.provider,
            "llm.default_model": d.llm.default_model,
            "llm.cost_budget_per_call": str(d.llm.cost_budget_per_call),
            "llm.anonymize_before_llm": d.llm.anonymize_before_llm,
            "llm.log_external_calls": d.llm.log_external_calls,
            "runtime.api.api_key": d.runtime.api.api_key,
            "runtime.api.cors_origins": d.runtime.api.cors_origins,
            "runtime.api.media_root": d.runtime.api.media_root,
            "runtime.api.rate_limit_rpm": str(d.runtime.api.rate_limit_rpm),
            "runtime.api.use_redis_cache": d.runtime.api.use_redis_cache,
            "runtime.api.redis_url": d.runtime.api.redis_url,
            "runtime.api.allow_client_llm_key": d.runtime.api.allow_client_llm_key,
            "runtime.dashboard.api_base_url": d.runtime.dashboard.api_base_url,
            "runtime.dashboard.dev_mode": d.runtime.dashboard.dev_mode,
            "runtime.alerting.webhook_enabled": d.runtime.alerting.webhook_enabled,
            "runtime.alerting.webhook_url": d.runtime.alerting.webhook_url,
            "runtime.alerting.timeout_seconds": str(d.runtime.alerting.timeout_seconds),
            "runtime.alerting.max_retries": str(d.runtime.alerting.max_retries),
            "runtime.alerting.circuit_breaker_threshold": str(d.runtime.alerting.circuit_breaker_threshold),
        }
        for key, val in mapping.items():
            if key not in self._vars:
                continue
            var = self._vars[key]
            if isinstance(var, tk.BooleanVar):
                var.set(bool(val))
            else:
                var.set(str(val))
        self._dirty = False

    def _collect_draft(self) -> UserConfig:
        def _s(key: str) -> str:
            return str(self._get_var(key).get()).strip()

        def _b(key: str) -> bool:
            return bool(self._get_var(key).get())

        def _i(key: str, default: int = 0) -> int:
            try:
                return int(_s(key))
            except ValueError:
                return default

        def _f(key: str, default: float = 0.0) -> float:
            try:
                return float(_s(key))
            except ValueError:
                return default

        updated = self._draft.model_copy(deep=True)
        updated.install_profile = InstallProfile(_s("install_profile"))
        updated.portable_mode = _b("portable_mode")
        updated.sentiment_profile = _s("sentiment_profile")
        updated.device = _s("device")  # type: ignore[assignment]
        updated.log_level = _s("log_level")  # type: ignore[assignment]
        updated.paths = updated.paths.model_copy(
            update={
                "outputs": _s("paths.outputs"),
                "hf_cache": _s("paths.hf_cache"),
                "llm_cache": _s("paths.llm_cache"),
                "logs": _s("paths.logs"),
                "incoming_calls": _s("paths.incoming_calls"),
                "state_dir": _s("paths.state_dir"),
                "app_root": str(self._app_root),
            }
        )
        updated.services = updated.services.model_copy(
            update={
                "api_enabled": _b("services.api_enabled"),
                "api_host": _s("services.api_host"),
                "api_port": _i("services.api_port", 8000),
                "dashboard_enabled": _b("services.dashboard_enabled"),
                "dashboard_port": _i("services.dashboard_port", 8080),
                "dashboard_ui": _s("services.dashboard_ui"),  # type: ignore[typeddict-item]
            }
        )
        updated.asr = updated.asr.model_copy(
            update={
                "backend": _s("asr.backend"),  # type: ignore[typeddict-item]
                "model": _s("asr.model"),
                "revision": _s("asr.revision"),  # type: ignore[typeddict-item]
                "language": _s("asr.language"),
                "hotwords_file": _s("asr.hotwords_file"),
                "preprocess_mode": _s("asr.preprocess_mode"),  # type: ignore[typeddict-item]
                "preprocess": _s("asr.preprocess_mode") != "off",
            }
        )
        updated.llm = updated.llm.model_copy(
            update={
                "enabled": _b("llm.enabled"),
                "provider": _s("llm.provider"),
                "default_model": _s("llm.default_model"),
                "cost_budget_per_call": _f("llm.cost_budget_per_call", 0.08),
                "anonymize_before_llm": _b("llm.anonymize_before_llm"),
                "log_external_calls": _b("llm.log_external_calls"),
            }
        )
        updated.runtime = updated.runtime.model_copy(
            update={
                "api": updated.runtime.api.model_copy(
                    update={
                        "api_key": _s("runtime.api.api_key"),
                        "cors_origins": _s("runtime.api.cors_origins"),
                        "media_root": _s("runtime.api.media_root"),
                        "rate_limit_rpm": _i("runtime.api.rate_limit_rpm"),
                        "use_redis_cache": _b("runtime.api.use_redis_cache"),
                        "redis_url": _s("runtime.api.redis_url"),
                        "allow_client_llm_key": _b("runtime.api.allow_client_llm_key"),
                    }
                ),
                "dashboard": updated.runtime.dashboard.model_copy(
                    update={
                        "api_base_url": _s("runtime.dashboard.api_base_url"),
                        "dev_mode": _b("runtime.dashboard.dev_mode"),
                    }
                ),
                "alerting": updated.runtime.alerting.model_copy(
                    update={
                        "webhook_enabled": _b("runtime.alerting.webhook_enabled"),
                        "webhook_url": _s("runtime.alerting.webhook_url"),
                        "timeout_seconds": _i("runtime.alerting.timeout_seconds", 10),
                        "max_retries": _i("runtime.alerting.max_retries", 3),
                        "circuit_breaker_threshold": _i(
                            "runtime.alerting.circuit_breaker_threshold", 5
                        ),
                    }
                ),
            }
        )
        return updated

    def _on_secret_typed(self, kind: SecretKind) -> None:
        entry = self._secret_entries[kind]
        self._pending_secrets[kind] = entry.get()
        self._dirty = True

    def _collect_all_pending_secrets(self) -> dict[SecretKind, str]:
        pending: dict[SecretKind, str] = {}
        for kind, entry in self._secret_entries.items():
            value = entry.get().strip()
            if value:
                pending[kind] = value
        return pending

    def _refresh_openrouter_status(self) -> None:
        info = secret_status(self._app_root).get("openrouter", {})
        if info.get("configured"):
            source = info.get("source", "unknown")
            path = user_secret_file("openrouter", self._app_root)
            self._llm_openrouter_status.set(f"Konfigurerad (källa: {source}, fil: {path})")
        else:
            self._llm_openrouter_status.set("Ingen nyckel sparad")

    def _save_openrouter_key_now(self) -> None:
        entry = self._secret_entries["openrouter"]
        value = entry.get().strip()
        if not value:
            messagebox.showwarning("OpenRouter", "Ange en API-nyckel först.", parent=self)
            return
        try:
            msg = save_secret_permanent("openrouter", value, self._app_root)
        except ValueError as exc:
            messagebox.showerror("OpenRouter", str(exc), parent=self)
            return
        self._pending_secrets.pop("openrouter", None)
        entry.delete(0, tk.END)
        self._refresh_openrouter_status()
        self._refresh_secret_status()
        self._event_log.info("OpenRouter key saved permanently", phase="settings")
        messagebox.showinfo("OpenRouter", msg, parent=self)

    def _save_secret_now(self, kind: SecretKind) -> None:
        entry = self._secret_entries[kind]
        value = entry.get().strip()
        if not value:
            messagebox.showwarning("Nycklar", "Ange ett värde först.", parent=self)
            return
        try:
            msg = save_secret_permanent(kind, value, self._app_root)
        except ValueError as exc:
            messagebox.showerror("Nycklar", str(exc), parent=self)
            return
        self._pending_secrets.pop(kind, None)
        entry.delete(0, tk.END)
        self._refresh_secret_status()
        self._event_log.info(f"Secret saved permanently: {kind}", phase="settings")
        messagebox.showinfo("Nycklar", msg, parent=self)

    def _refresh_secret_status(self) -> None:
        status = secret_status(self._app_root)
        lines = []
        for kind in _SECRET_KINDS:
            info = status.get(kind, {})
            configured = "ja" if info.get("configured") else "nej"
            source = info.get("source", "none")
            lines.append(f"{kind}: konfigurerad={configured}, källa={source}")
        self._secret_status_var.set("\n".join(lines))

    def _clear_secret(self, kind: SecretKind) -> None:
        clear_secret(kind, self._app_root)
        self._pending_secrets.pop(kind, None)
        self._secret_entries[kind].delete(0, tk.END)
        self._refresh_secret_status()
        if kind == "openrouter":
            self._refresh_openrouter_status()
        self._event_log.info(f"Cleared secret: {kind}", phase="settings")

    def _save(self) -> None:
        draft = self._collect_draft()
        pending = self._collect_all_pending_secrets() or None
        issues = validate_draft(draft, pending_secrets=pending)
        if issues:
            messagebox.showerror(
                "Validering",
                "\n".join(f"{i.field}: {i.message}" for i in issues),
                parent=self,
            )
            return
        try:
            result = save_draft(
                draft,
                baseline=self._baseline,
                pending_secrets=pending,
            )
        except ValueError as exc:
            messagebox.showerror("Spara", str(exc), parent=self)
            return

        self._baseline = draft.model_copy(deep=True)
        self._draft = draft
        self._dirty = False
        self._pending_secrets.clear()
        for entry in self._secret_entries.values():
            entry.delete(0, tk.END)
        self._refresh_secret_status()
        self._refresh_openrouter_status()
        self._event_log.info(f"Settings saved to {result.path}", phase="settings")

        msg = f"Sparat till:\n{result.path}"
        if result.secrets_saved:
            parts = []
            for kind, backends in result.secrets_saved.items():
                parts.append(f"{kind}: {', '.join(backends)}")
            msg += "\n\nNycklar sparade:\n" + "\n".join(parts)
        if result.restart_services:
            msg += f"\n\nStarta om: {', '.join(result.restart_services)}"
        if result.profile_changed:
            msg += "\n\nInstallationsprofil ändrad — kör Installera/Reparera vid behov."
        messagebox.showinfo("Inställningar", msg, parent=self)

        if self._on_saved:
            self._on_saved()

    def _on_close(self) -> None:
        if self._dirty and not messagebox.askyesno(
            "Inställningar", "Osparade ändringar. Stäng ändå?", parent=self
        ):
            return
        self.grab_release()
        self.destroy()

    def _trigger_provision(self) -> None:
        if self._on_provision:
            self._on_provision()

    def _open_asr_manager(self) -> None:
        if self._on_open_asr:
            self._on_open_asr()
        else:
            open_asr_manager_dialog(self.master, self._collect_draft(), self._event_log)

    def _export_config(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Exportera config",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        export_bundle(self._collect_draft(), Path(path))
        messagebox.showinfo("Export", f"Exporterad till {path}", parent=self)

    def _import_config(self) -> None:
        path = filedialog.askopenfilename(
            parent=self, title="Importera config", filetypes=[("JSON", "*.json")]
        )
        if not path:
            return
        try:
            cfg = import_bundle(Path(path), self._app_root)
        except Exception as exc:
            messagebox.showerror("Import", str(exc), parent=self)
            return
        self._draft = cfg
        self._populate_from_draft()
        self._dirty = True
        messagebox.showinfo("Import", "Config importerad — granska och spara.", parent=self)

    def _run_doctor(self) -> None:
        lines = []
        for ok, name, msg in run_doctor(self._collect_draft()):
            icon = "OK" if ok else "FAIL"
            lines.append(f"[{icon}] {name}: {msg}")
        self._doctor_text.configure(state=tk.NORMAL)
        self._doctor_text.delete("1.0", tk.END)
        self._doctor_text.insert(tk.END, "\n".join(lines))
        self._doctor_text.configure(state=tk.DISABLED)


def open_settings_dialog(
    master: tk.Misc,
    app_root: Path,
    event_log: EventLog,
    *,
    on_saved: Callable[[], None] | None = None,
    on_provision: Callable[[], None] | None = None,
    on_open_asr: Callable[[], None] | None = None,
) -> None:
    """Open the settings dialog."""
    SettingsDialog(
        master,
        app_root,
        event_log,
        on_saved=on_saved,
        on_provision=on_provision,
        on_open_asr=on_open_asr,
    )