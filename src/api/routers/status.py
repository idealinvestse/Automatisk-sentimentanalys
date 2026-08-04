"""Process status and detailed health endpoints for observability."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request

from ...analysis.registry import ensure_analyzers_loaded, get_analyzer_registry
from ...core.status import derive_job_status, get_status_reporter
from ...transcription.factory import list_available_backends

router = APIRouter(prefix="/status", tags=["Status"])


@router.get("/processes")
async def list_process_events(
    limit: int = Query(100, ge=1, le=1000),
    job_id: str | None = Query(None),
    component: str | None = Query(None),
    level: str | None = Query(None),
    since: str | None = Query(None, description="ISO timestamp lower bound"),
) -> dict[str, Any]:
    """Return recent process status events (filterable for ops dashboard)."""
    reporter = get_status_reporter()
    events = reporter.recent_events(
        limit=limit,
        job_id=job_id,
        component=component,
        level=level,
        since=since,
    )
    return {"events": events, "count": len(events)}


@router.get("/jobs/{job_id}")
async def job_status(job_id: str) -> dict[str, Any]:
    """Live status summary for a single job_id."""
    reporter = get_status_reporter()
    events = reporter.recent_events(limit=1000, job_id=job_id)
    return derive_job_status(events, job_id)


@router.get("/health/detail")
async def health_detail(request: Request) -> dict[str, Any]:
    """Extended health with component availability and degraded signals."""
    from ..settings import get_api_settings

    ensure_analyzers_loaded()
    analyzers = sorted(get_analyzer_registry().keys())
    settings = get_api_settings()
    cache = getattr(request.app.state, "cache", None)
    cache_stats: dict[str, Any] = {}
    if cache is not None and hasattr(cache, "stats"):
        try:
            cache_stats = cache.stats()  # type: ignore[call-arg]
        except Exception:
            cache_stats = {"available": True}
    elif cache is not None:
        cache_stats = {"available": True}

    asr_backends = list_available_backends()
    hub = getattr(request.app.state, "transcription_events", None)
    tickets = getattr(request.app.state, "ws_tickets", None)
    redis_client = getattr(cache, "redis_client", None) if cache is not None else None
    redis_ok = redis_client is not None
    if redis_ok:
        try:
            redis_client.ping()
        except Exception:
            redis_ok = False

    checks: dict[str, bool] = {
        "analyzers_loaded": len(analyzers) > 0,
        "asr_backend_available": len(asr_backends) > 0,
        "media_root_configured": bool(settings.media_root),
        "redis_configured": bool(settings.use_redis_cache),
        "redis_reachable": redis_ok if settings.use_redis_cache else True,
    }
    degraded_reasons: list[str] = []
    if not checks["analyzers_loaded"]:
        degraded_reasons.append("no_analyzers")
    if not checks["asr_backend_available"]:
        degraded_reasons.append("no_asr_backend")
    if settings.production and not checks["media_root_configured"]:
        degraded_reasons.append("media_root_missing")
    if settings.use_redis_cache and not checks["redis_reachable"]:
        degraded_reasons.append("redis_unreachable")
    if hub is not None and getattr(hub, "backend", "memory") == "memory" and settings.use_redis_cache:
        degraded_reasons.append("ws_hub_memory_fallback")

    status = "degraded" if degraded_reasons else "ok"
    return {
        "status": status,
        "checks": checks,
        "degraded": degraded_reasons,
        "analyzers": {
            "count": len(analyzers),
            "registered": analyzers,
        },
        "asr": {
            "backends": asr_backends,
        },
        "cache": cache_stats,
        "transcription_events": {
            "backend": getattr(hub, "backend", "unknown") if hub is not None else "unavailable",
        },
        "ws_tickets": {
            "backend": getattr(tickets, "backend", "unknown") if tickets is not None else "unavailable",
        },
        "recent_events": get_status_reporter().recent_events(limit=5),
    }
