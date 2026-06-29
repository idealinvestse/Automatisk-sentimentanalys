"""Process status and detailed health endpoints for observability."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request

from ...analysis.registry import ensure_analyzers_loaded, get_analyzer_registry
from ...core.status import get_status_reporter
from ...transcription.factory import list_available_backends

router = APIRouter(prefix="/status", tags=["Status"])


@router.get("/processes")
async def list_process_events(
    limit: int = Query(100, ge=1, le=1000),
) -> dict[str, Any]:
    """Return recent process status events (for debugging and future ops dashboard)."""
    reporter = get_status_reporter()
    events = reporter.recent_events(limit=limit)
    return {"events": events, "count": len(events)}


@router.get("/health/detail")
async def health_detail(request: Request) -> dict[str, Any]:
    """Extended health with component availability."""
    ensure_analyzers_loaded()
    analyzers = sorted(get_analyzer_registry().keys())
    cache = getattr(request.app.state, "cache", None)
    cache_stats: dict[str, Any] = {}
    if cache is not None and hasattr(cache, "stats"):
        try:
            cache_stats = cache.stats()  # type: ignore[call-arg]
        except Exception:
            cache_stats = {"available": True}
    elif cache is not None:
        cache_stats = {"available": True}

    return {
        "status": "ok",
        "analyzers": {
            "count": len(analyzers),
            "registered": analyzers,
        },
        "asr": {
            "backends": list_available_backends(),
        },
        "cache": cache_stats,
        "recent_events": get_status_reporter().recent_events(limit=5),
    }
