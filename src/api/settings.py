"""API runtime settings from environment (Fas 2 hardening)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


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
    rate_limit_rpm: int

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key)


def _runtime_api_defaults() -> dict[str, object]:
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
        rate_limit_rpm=rate_limit,
    )
