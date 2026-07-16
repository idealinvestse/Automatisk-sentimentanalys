"""Load, merge, and save user configuration YAML."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .config_schema import UserConfig
from .paths_util import resolve_user_config_path

_DEFAULTS_ENV = "SENTIMENT_INSTALL_DEFAULTS"


def default_user_config_path(portable: bool = False, app_root: Path | None = None) -> Path:
    root = (app_root or Path.cwd()).resolve()
    return resolve_user_config_path(root, portable=portable if portable else None)


def install_defaults_path(app_root: Path | None = None) -> Path:
    override = os.environ.get(_DEFAULTS_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    root = app_root or Path.cwd()
    return root / "configs" / "install_defaults.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_yaml_dict(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def merge_configs(
    defaults: dict[str, Any] | None = None,
    user: dict[str, Any] | None = None,
) -> UserConfig:
    merged = _deep_merge(defaults or {}, user or {})
    return UserConfig.model_validate(merged)


def load_user_config(
    app_root: Path | None = None,
    *,
    portable: bool | None = None,
    create_if_missing: bool = False,
    persist_healed: bool = True,
) -> UserConfig:
    root = (app_root or Path.cwd()).resolve()
    defaults = load_yaml_dict(install_defaults_path(root))
    user_path = resolve_user_config_path(root, portable=portable)
    user_data = load_yaml_dict(user_path)

    if create_if_missing and not user_path.is_file():
        use_portable = portable if portable is not None else (root / "user_data").is_dir()
        user_path = resolve_user_config_path(root, portable=use_portable)
        cfg = merge_configs(
            defaults,
            {
                **user_data,
                "paths": {"app_root": str(root)},
                "portable_mode": bool(use_portable),
            },
        )
        save_user_config(cfg, path=user_path)
        return cfg

    merged = merge_configs(defaults, user_data)
    if not merged.paths.app_root:
        merged.paths.app_root = str(root)
    if portable is not None:
        merged.portable_mode = portable
    else:
        from .paths_util import portable_user_config_path

        merged.portable_mode = user_path == portable_user_config_path(root) or bool(
            merged.portable_mode
        )

    from .paths_util import heal_app_root, migrate_legacy_dashboard_settings

    configured = Path(merged.paths.app_root)
    healed = heal_app_root(configured, preferred=root)
    changed = False
    if healed != configured.resolve():
        merged.paths.app_root = str(healed)
        changed = True
    if migrate_legacy_dashboard_settings(merged):
        changed = True
    if changed and persist_healed and user_path.is_file():
        save_user_config(merged, path=user_path)
    return merged


def save_user_config(cfg: UserConfig, path: Path | None = None) -> Path:
    target = path or default_user_config_path(
        portable=cfg.portable_mode,
        app_root=cfg.resolved_app_root(),
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = cfg.model_dump(mode="json")
    with target.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    return target


def _connect_host_for_clients(bind_host: str) -> str:
    """Map bind-all addresses to loopback for browser/client URLs."""
    normalized = bind_host.strip().lower()
    if normalized in ("0.0.0.0", "::", "[::]"):
        return "127.0.0.1"
    return bind_host.strip() or "127.0.0.1"


def derive_local_api_base_url(cfg: UserConfig) -> str:
    """Local API base URL derived from services.api_host/api_port."""
    host = _connect_host_for_clients(cfg.services.api_host)
    return f"http://{host}:{cfg.services.api_port}"


def effective_dashboard_api_base_url(cfg: UserConfig) -> str:
    """Dashboard → API URL: configured value, or derived from services ports."""
    configured = (cfg.runtime.dashboard.api_base_url or "").strip()
    return configured or derive_local_api_base_url(cfg)


def sync_api_base_url_from_services(
    draft: UserConfig,
    *,
    baseline: UserConfig | None = None,
) -> bool:
    """Keep runtime.dashboard.api_base_url aligned with api host/port when safe.

    Updates when the URL is empty, or still matches the previous locally derived
    URL (user had not customized it). Preserves remote/custom URLs.
    Returns True if ``draft`` was modified.
    """
    derived_new = derive_local_api_base_url(draft)
    current = (draft.runtime.dashboard.api_base_url or "").strip().rstrip("/")

    if baseline is None:
        if not current:
            draft.runtime.dashboard.api_base_url = derived_new
            return True
        return False

    host_or_port_changed = (
        baseline.services.api_host != draft.services.api_host
        or baseline.services.api_port != draft.services.api_port
    )
    if not host_or_port_changed:
        return False

    derived_old = derive_local_api_base_url(baseline).rstrip("/")
    if not current or current == derived_old:
        draft.runtime.dashboard.api_base_url = derived_new
        return True
    return False


def derive_local_dashboard_origins(cfg: UserConfig) -> list[str]:
    """Browser origins for the local Next.js dashboard (localhost + 127.0.0.1)."""
    port = int(cfg.services.dashboard_port or 3000)
    return [f"http://localhost:{port}", f"http://127.0.0.1:{port}"]


def _cors_origin_set(csv: str) -> set[str]:
    return {o.strip().rstrip("/") for o in csv.split(",") if o.strip()}


def effective_cors_origins_csv(cfg: UserConfig) -> str | None:
    """CORS list for the API child: explicit config, else local dashboard origins."""
    configured = (cfg.runtime.api.cors_origins or "").strip()
    if configured:
        return configured
    if cfg.services.dashboard_enabled:
        return ",".join(derive_local_dashboard_origins(cfg))
    return None


def sync_cors_origins_from_dashboard(
    draft: UserConfig,
    *,
    baseline: UserConfig | None = None,
) -> bool:
    """Keep runtime.api.cors_origins aligned with dashboard_port when safe.

    Updates when CORS is empty, or still equals the previous local dashboard
    defaults. Preserves custom origin lists.
    """
    derived_new = ",".join(derive_local_dashboard_origins(draft))
    current = (draft.runtime.api.cors_origins or "").strip()

    if baseline is None:
        if not current and draft.services.dashboard_enabled:
            draft.runtime.api.cors_origins = derived_new
            return True
        return False

    if baseline.services.dashboard_port == draft.services.dashboard_port:
        return False

    derived_old = set(derive_local_dashboard_origins(baseline))
    current_set = _cors_origin_set(current)
    if not current_set or current_set == {o.rstrip("/") for o in derived_old}:
        draft.runtime.api.cors_origins = derived_new
        return True
    return False


def config_to_env(cfg: UserConfig) -> dict[str, str]:
    """Environment variables for child processes (API, CLI, dashboard)."""
    env: dict[str, str] = {}
    env["HF_HOME"] = str(cfg.resolved_hf_home())
    env["HUGGINGFACE_HUB_CACHE"] = str(cfg.resolved_hf_home() / "hub")
    env["SENTIMENT_LLM_CACHE"] = str(cfg.resolved_llm_cache())
    env["SENTIMENT_OUTPUTS"] = str(cfg.resolved_outputs())
    env["SENTIMENT_APP_ROOT"] = str(cfg.resolved_app_root())
    env["SENTIMENT_USER_DATA"] = str(cfg.resolved_user_data_dir())
    if cfg.paths.data_root.strip():
        env["SENTIMENT_DATA_ROOT"] = str(cfg.resolved_data_root())
    env["SENTIMENT_LOG_LEVEL"] = cfg.log_level
    env["LOG_LEVEL"] = cfg.log_level
    if cfg.llm.enabled:
        env["SENTIMENT_LLM_ENABLED"] = "1"

    api_rt = cfg.runtime.api
    if api_rt.api_key:
        env["SENTIMENT_API_KEY"] = api_rt.api_key
        # Pilot/LAN: webui browser client reads NEXT_PUBLIC_API_KEY.
        env["NEXT_PUBLIC_API_KEY"] = api_rt.api_key
    cors_csv = effective_cors_origins_csv(cfg)
    if cors_csv:
        env["API_CORS_ORIGINS"] = cors_csv
    if api_rt.media_root:
        env["API_MEDIA_ROOT"] = api_rt.media_root
    if api_rt.rate_limit_rpm > 0:
        env["API_RATE_LIMIT_RPM"] = str(api_rt.rate_limit_rpm)
    if api_rt.use_redis_cache:
        env["API_USE_REDIS_CACHE"] = "1"
    if api_rt.redis_url:
        env["REDIS_URL"] = api_rt.redis_url
    if api_rt.allow_client_llm_key:
        env["API_ALLOW_CLIENT_LLM_KEY"] = "1"

    alert = cfg.runtime.alerting
    if alert.webhook_enabled and alert.webhook_url:
        env["ALERT_WEBHOOK_URL"] = alert.webhook_url
    env["ALERT_WEBHOOK_TIMEOUT"] = str(alert.timeout_seconds)
    env["ALERT_WEBHOOK_RETRIES"] = str(alert.max_retries)
    env["ALERT_WEBHOOK_BREAKER"] = str(alert.circuit_breaker_threshold)
    env["ALERT_WEBHOOK_BACKOFF"] = str(alert.retry_backoff_base)

    dash = cfg.runtime.dashboard
    api_base = effective_dashboard_api_base_url(cfg)
    env["SENTIMENT_API_BASE_URL"] = api_base
    # Next.js webui reads NEXT_PUBLIC_* (not SENTIMENT_*); keep both in sync.
    env["NEXT_PUBLIC_API_BASE_URL"] = api_base
    if dash.storage_secret:
        env["DASHBOARD_STORAGE_SECRET"] = dash.storage_secret
    if dash.dev_mode:
        env["SENTIMENT_DEV_MODE"] = "1"

    return env
