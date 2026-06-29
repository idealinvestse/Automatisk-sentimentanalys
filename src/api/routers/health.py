"""Health check and metrics router."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response

from ...alerting_state import AlertingStateManager
from ..dependencies import require_api_key
from ..metrics import render_metrics, update_alerting_metrics

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health() -> dict[str, str]:
    """Simple health check endpoint.

    Returns:
        ``{"status": "ok"}`` when the service is running.
    """
    return {"status": "ok"}


@router.get("/metrics", dependencies=[Depends(require_api_key)])
async def metrics(request: Request) -> Response:
    """Prometheus metrics endpoint (requires ``X-API-Key`` when auth is enabled)."""
    state = getattr(request.app.state, "alerting_state", None)
    if isinstance(state, AlertingStateManager):
        update_alerting_metrics(state)
    body, content_type = render_metrics()
    return Response(content=body, media_type=content_type)
