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


@lru_cache
def get_api_settings() -> APISettings:
    cors_raw = os.getenv("API_CORS_ORIGINS", "")
    origins = [o.strip() for o in cors_raw.split(",") if o.strip()]
    return APISettings(
        api_key=os.getenv("SENTIMENT_API_KEY") or None,
        cors_origins=origins,
        allow_client_llm_key_in_body=os.getenv("API_ALLOW_CLIENT_LLM_KEY", "false").lower()
        in ("1", "true", "yes"),
        media_root=os.getenv("API_MEDIA_ROOT") or None,
        use_redis_cache=os.getenv("API_USE_REDIS_CACHE", "false").lower() in ("1", "true", "yes"),
        redis_url=os.getenv("REDIS_URL"),
        cache_dir=os.getenv("API_CACHE_DIR", ".cache/aggregates"),
        rate_limit_rpm=int(os.getenv("API_RATE_LIMIT_RPM", "0") or "0"),
    )
