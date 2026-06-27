"""Launcher settings load/save/validate (UI-agnostic)."""

from __future__ import annotations

import json
import secrets
import socket
from dataclasses import dataclass, field
from pathlib import Path
from pydantic import ValidationError

from src.install.config_schema import UserConfig
from src.install.preflight import run_preflight
from src.install.secrets_win import SecretKind, delete_secret, get_secret, secret_status, set_secret
from src.install.user_config import load_user_config, save_user_config

_SENTIMENT_PROFILES = (
    "default",
    "forum",
    "call",
    "callcenter",
    "news",
    "social",
    "review",
    "magazine",
    "sales",
    "complaint",
    "support",
    "teknisk_support",
)
_DEVICES = ("auto", "cpu", "cuda", "cuda:0", "mps")
_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
_SECRET_KINDS: tuple[SecretKind, ...] = ("openrouter", "huggingface", "groq")


@dataclass
class ValidationIssue:
    field: str
    message: str


@dataclass
class SettingsSnapshot:
    config: UserConfig
    config_path: Path
    secrets: dict[str, dict[str, bool | str]]
    baseline: UserConfig | None = None


@dataclass
class SaveResult:
    path: Path
    restart_services: list[str] = field(default_factory=list)
    profile_changed: bool = False


def load_settings(app_root: Path | None = None) -> SettingsSnapshot:
    cfg = load_user_config(app_root)
    from src.install.user_config import default_user_config_path

    config_path = default_user_config_path(portable=cfg.portable_mode, app_root=cfg.resolved_app_root())
    return SettingsSnapshot(
        config=cfg,
        config_path=config_path,
        secrets=secret_status(cfg.resolved_app_root()),
    )


def build_draft(cfg: UserConfig) -> UserConfig:
    return cfg.model_copy(deep=True)


def _port_available(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            return sock.connect_ex((host, port)) != 0
    except OSError:
        return True


def validate_draft(
    draft: UserConfig,
    *,
    pending_secrets: dict[SecretKind, str] | None = None,
    check_ports: bool = True,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    try:
        UserConfig.model_validate(draft.model_dump(mode="json"))
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", ()))
            issues.append(ValidationIssue(loc or "config", err.get("msg", "validation error")))
        return issues

    if draft.services.api_port == draft.services.dashboard_port:
        issues.append(
            ValidationIssue("services", "API-port och dashboard-port måste vara olika.")
        )

    if check_ports and draft.services.api_enabled:
        if not _port_available(draft.services.api_host, draft.services.api_port):
            issues.append(
                ValidationIssue(
                    "services.api_port",
                    f"Port {draft.services.api_port} verkar upptagen på {draft.services.api_host}.",
                )
            )
    if check_ports and draft.services.dashboard_enabled:
        if not _port_available("127.0.0.1", draft.services.dashboard_port):
            issues.append(
                ValidationIssue(
                    "services.dashboard_port",
                    f"Port {draft.services.dashboard_port} verkar upptagen.",
                )
            )

    if draft.llm.enabled:
        has_key = bool(pending_secrets and pending_secrets.get("openrouter"))
        if draft.llm.provider == "openrouter" and not has_key and not get_secret("openrouter"):
            issues.append(
                ValidationIssue("llm", "LLM aktiverat med OpenRouter men ingen API-nyckel konfigurerad.")
            )
        if draft.llm.provider == "groq" and not (
            (pending_secrets and pending_secrets.get("groq")) or get_secret("groq")
        ):
            issues.append(
                ValidationIssue("llm", "LLM aktiverat med Groq men ingen API-nyckel konfigurerad.")
            )

    if draft.runtime.alerting.webhook_enabled and not draft.runtime.alerting.webhook_url.strip():
        issues.append(
            ValidationIssue("runtime.alerting.webhook_url", "Webhook aktiverad men URL saknas.")
        )

    return issues


def _ensure_dashboard_secret(draft: UserConfig) -> None:
    if not draft.runtime.dashboard.storage_secret:
        draft.runtime.dashboard.storage_secret = secrets.token_urlsafe(32)


def restart_hints(before: UserConfig, after: UserConfig) -> list[str]:
    services: list[str] = []
    if before.services.api_port != after.services.api_port or before.services.api_host != after.services.api_host:
        services.append("api")
    if before.services.dashboard_port != after.services.dashboard_port:
        services.append("dashboard")
    runtime_before = before.runtime.model_dump()
    runtime_after = after.runtime.model_dump()
    if runtime_before != runtime_after:
        if "api" not in services:
            services.append("api")
        if "dashboard" not in services:
            services.append("dashboard")
    if before.install_profile != after.install_profile:
        if "api" not in services:
            services.append("api")
        if "dashboard" not in services:
            services.append("dashboard")
    return services


def save_draft(
    draft: UserConfig,
    *,
    baseline: UserConfig | None = None,
    pending_secrets: dict[SecretKind, str] | None = None,
    check_ports: bool = True,
) -> SaveResult:
    issues = validate_draft(draft, pending_secrets=pending_secrets, check_ports=check_ports)
    if issues:
        raise ValueError("; ".join(f"{i.field}: {i.message}" for i in issues))

    _ensure_dashboard_secret(draft)
    if not draft.paths.app_root:
        draft.paths.app_root = str(draft.resolved_app_root())

    path = save_user_config(draft)

    if pending_secrets:
        for kind, value in pending_secrets.items():
            if value.strip():
                set_secret(kind, value)

    profile_changed = bool(baseline and baseline.install_profile != draft.install_profile)
    services = restart_hints(baseline, draft) if baseline else []
    return SaveResult(path=path, restart_services=services, profile_changed=profile_changed)


def export_bundle(cfg: UserConfig, path: Path) -> None:
    bundle = {"config": cfg.model_dump(mode="json")}
    path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")


def import_bundle(path: Path, app_root: Path) -> UserConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    cfg = UserConfig.model_validate(data.get("config", data))
    if not cfg.paths.app_root:
        cfg.paths.app_root = str(app_root.resolve())
    save_user_config(cfg)
    return cfg


def run_doctor(cfg: UserConfig) -> list[tuple[bool, str, str]]:
    report = run_preflight(cfg, require_openrouter=cfg.llm.enabled and cfg.llm.provider == "openrouter")
    return [(c.ok, c.name, c.message + (f" — {c.detail}" if c.detail else "")) for c in report.checks]


def clear_secret(kind: SecretKind) -> None:
    delete_secret(kind)