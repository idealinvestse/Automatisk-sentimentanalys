"""API runtime settings from environment (Fas 2 hardening)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


@dataclass(frozen=True)
class APISettings:
    """Configuration for REST API security and infrastructure."""

    api_key: str | None
    cors_origins: list[str]
    allow_client_llm_key_in_body: bool
    media_root: str | None
    use_redis_cache: bool
    redis_url: str | None
    cache_dir: str
    state_dir: str
    rate_limit_rpm: int
    trusted_proxy: bool
    production: bool
    require_auth: bool
    require_media_root: bool
    max_upload_size_mb: int
    upload_retention_days: int

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key)


def validate_production_settings(settings: APISettings) -> None:
    """Fail fast when production guards are enabled but misconfigured.

    ``API_PRODUCTION=true`` implies auth + media-root sandbox even if the
    individual require_* flags are left at defaults.
    """
    from ..core.errors import ConfigurationError

    need_auth = settings.production or settings.require_auth
    need_media = settings.production or settings.require_media_root
    if need_auth and not settings.api_key:
        raise ConfigurationError(
            "SENTIMENT_API_KEY is required when API_PRODUCTION or API_REQUIRE_AUTH is set"
        )
    if need_media and not settings.media_root:
        raise ConfigurationError(
            "API_MEDIA_ROOT is required when API_PRODUCTION or API_REQUIRE_MEDIA_ROOT is set"
        )
    if settings.production and settings.use_redis_cache and not settings.redis_url:
        raise ConfigurationError(
            "REDIS_URL is required when API_PRODUCTION and API_USE_REDIS_CACHE are both set"
        )


def _runtime_api_defaults() -> dict[str, Any]:
    """Fallback to user_config.yaml when env vars are unset (launcher-managed)."""
    try:
        from ..install.user_config import load_user_config

        rt = load_user_config().runtime.api
        return {
            "api_key": rt.api_key or None,
            "cors_origins": rt.cors_origins,
            "media_root": rt.media_root or None,
            "use_redis_cache": rt.use_redis_cache,
            "redis_url": rt.redis_url or None,
            "rate_limit_rpm": rt.rate_limit_rpm,
            "allow_client_llm_key": rt.allow_client_llm_key,
        }
    except Exception:
        return {}


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes")


@lru_cache
def get_api_settings() -> APISettings:
    defaults = _runtime_api_defaults()
    cors_raw = os.getenv("API_CORS_ORIGINS")
    if cors_raw is None:
        cors_raw = str(defaults.get("cors_origins") or "")
    origins = [o.strip() for o in cors_raw.split(",") if o.strip()]
    api_key = os.getenv("SENTIMENT_API_KEY")
    if api_key is None:
        api_key = defaults.get("api_key")
    rate_env = os.getenv("API_RATE_LIMIT_RPM")
    rate_limit = int(rate_env) if rate_env is not None else int(defaults.get("rate_limit_rpm") or 0)
    max_upload_env = os.getenv("API_MAX_UPLOAD_SIZE_MB")
    max_upload = int(max_upload_env) if max_upload_env is not None else 200
    retention_env = os.getenv("API_UPLOAD_RETENTION_DAYS")
    retention = int(retention_env) if retention_env is not None else 7
    return APISettings(
        api_key=api_key or None,
        cors_origins=origins,
        allow_client_llm_key_in_body=_env_bool(
            "API_ALLOW_CLIENT_LLM_KEY", bool(defaults.get("allow_client_llm_key"))
        ),
        media_root=os.getenv("API_MEDIA_ROOT") or defaults.get("media_root"),
        use_redis_cache=_env_bool("API_USE_REDIS_CACHE", bool(defaults.get("use_redis_cache"))),
        redis_url=os.getenv("REDIS_URL") or defaults.get("redis_url"),
        cache_dir=os.getenv("API_CACHE_DIR", ".cache/aggregates"),
        state_dir=os.getenv("API_STATE_DIR")
        or os.getenv("API_CACHE_DIR", ".cache/aggregates")
        or ".cache/aggregates",
        rate_limit_rpm=rate_limit,
        trusted_proxy=_env_bool("API_TRUSTED_PROXY"),
        production=_env_bool("API_PRODUCTION"),
        require_auth=_env_bool("API_REQUIRE_AUTH"),
        max_upload_size_mb=max_upload,
        upload_retention_days=retention,
        require_media_root=_env_bool("API_REQUIRE_MEDIA_ROOT"),
    )
