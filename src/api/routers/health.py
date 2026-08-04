"""Health check and metrics router."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse

from ...alerting_state import AlertingStateManager
from ..dependencies import require_api_key
from ..metrics import render_metrics, update_alerting_metrics
from ..settings import get_api_settings

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe — process is up (no dependency checks)."""
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request) -> JSONResponse:
    """Readiness probe — fail when production-critical deps are missing."""
    settings = get_api_settings()
    checks: dict[str, Any] = {
        "auth_configured": bool(settings.api_key) if settings.production else True,
        "media_root": bool(settings.media_root) if (settings.production or settings.require_media_root) else True,
    }
    cache = getattr(request.app.state, "cache", None)
    if settings.use_redis_cache:
        redis_client = getattr(cache, "redis_client", None) if cache is not None else None
        redis_ok = False
        if redis_client is not None:
            try:
                redis_ok = bool(redis_client.ping())
            except Exception:
                redis_ok = False
        checks["redis"] = redis_ok
    ready_ok = all(checks.values())
    body = {"status": "ready" if ready_ok else "not_ready", "checks": checks}
    return JSONResponse(content=body, status_code=200 if ready_ok else 503)


@router.get("/metrics", dependencies=[Depends(require_api_key)])
async def metrics(request: Request) -> Response:
    """Prometheus metrics endpoint (requires ``X-API-Key`` when auth is enabled)."""
    state = getattr(request.app.state, "alerting_state", None)
    if isinstance(state, AlertingStateManager):
        update_alerting_metrics(state)
    body, content_type = render_metrics()
    return Response(content=body, media_type=content_type)
