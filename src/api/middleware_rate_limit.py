"""Simple in-memory rate limiting middleware (Fas 3)."""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .error_responses import ERROR_CODE_RATE_LIMITED, build_error_content

logger = logging.getLogger(__name__)

_EXEMPT_PATHS = frozenset({"/health", "/docs", "/openapi.json", "/redoc"})


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-client sliding-window rate limiter (disabled when limit is 0)."""

    def __init__(self, app, *, requests_per_minute: int = 0) -> None:  # type: ignore[no-untyped-def]
        super().__init__(app)
        self.requests_per_minute = max(0, requests_per_minute)
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def _client_key(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        if request.client:
            return request.client.host
        return "unknown"

    def _allow(self, key: str) -> bool:
        if self.requests_per_minute <= 0:
            return True
        now = time.monotonic()
        window = 60.0
        q = self._hits[key]
        while q and now - q[0] > window:
            q.popleft()
        if len(q) >= self.requests_per_minute:
            return False
        q.append(now)
        return True

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        if self.requests_per_minute <= 0 or request.url.path in _EXEMPT_PATHS:
            return await call_next(request)
        key = self._client_key(request)
        if not self._allow(key):
            logger.warning("Rate limit exceeded for %s on %s", key, request.url.path)
            request_id = getattr(request.state, "request_id", None) or request.headers.get(
                "X-Request-ID"
            )
            return JSONResponse(
                status_code=429,
                content=build_error_content(
                    "Rate limit exceeded. Try again later.",
                    request_id=request_id,
                    error_code=ERROR_CODE_RATE_LIMITED,
                ),
                headers={"X-Request-ID": request_id} if request_id else {},
            )
        return await call_next(request)
