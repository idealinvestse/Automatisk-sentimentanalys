"""Alerting status and management endpoints.

Exposes webhook/circuit breaker health for dashboard and ops.
Uses ``app.state.alert_engine`` from lifespan for per-worker consistency.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from ..router_errors import run_route_sync

router = APIRouter(prefix="/alerting", tags=["alerting"])


def _get_engine(request: Request):
    return request.app.state.alert_engine


@router.get("/status", summary="Get current webhook and circuit breaker status")
def get_alerting_status(request: Request) -> dict:
    """Return current alerting/webhook health."""

    def _run() -> dict:
        engine = _get_engine(request)
        status = engine.get_webhook_status()
        return {
            "ok": True,
            "webhook": status,
            "note": "Circuit breaker state is per worker (app.state.alert_engine). "
            "Use AlertingStateManager for cross-worker JSON persistence.",
        }

    return run_route_sync("GET /alerting/status", _run)


@router.post("/reset-circuit-breaker", summary="Reset the webhook circuit breaker")
def reset_circuit_breaker(request: Request) -> dict:
    """Manually reset the circuit breaker (ops / testing)."""

    def _run() -> dict:
        engine = _get_engine(request)
        engine.reset_circuit_breaker()
        return {
            "ok": True,
            "message": "Circuit breaker has been reset.",
            "new_status": engine.get_webhook_status(),
        }

    return run_route_sync("POST /alerting/reset-circuit-breaker", _run)
