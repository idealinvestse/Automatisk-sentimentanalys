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


def config_to_env(cfg: UserConfig) -> dict[str, str]:
    """Environment variables for child processes (API, CLI, dashboard)."""
    env: dict[str, str] = {}
    env["HF_HOME"] = str(cfg.resolved_hf_home())
    env["SENTIMENT_APP_ROOT"] = str(cfg.resolved_app_root())
    env["SENTIMENT_USER_DATA"] = str(cfg.resolved_user_data_dir())
    env["SENTIMENT_LOG_LEVEL"] = cfg.log_level
    if cfg.llm.enabled:
        env["SENTIMENT_LLM_ENABLED"] = "1"

    api_rt = cfg.runtime.api
    if api_rt.api_key:
        env["SENTIMENT_API_KEY"] = api_rt.api_key
    if api_rt.cors_origins:
        env["API_CORS_ORIGINS"] = api_rt.cors_origins
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
    if dash.api_base_url:
        env["SENTIMENT_API_BASE_URL"] = dash.api_base_url
    if dash.storage_secret:
        env["NICEGUI_STORAGE_SECRET"] = dash.storage_secret
    if dash.dev_mode:
        env["SENTIMENT_DEV_MODE"] = "1"

    return env
