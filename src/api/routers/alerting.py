"""Alerting status and management endpoints.

Exposes webhook/circuit breaker health for dashboard and ops.

Note on multi-worker: Circuit breaker state is currently per-process.
For true multi-worker consistency, consider Redis or a central store (see TASK-07).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from src.alerting import AlertEngine

router = APIRouter(prefix="/alerting", tags=["alerting"])


def _get_engine(request: Request | None = None) -> AlertEngine:
    """Get AlertEngine instance.

    Priority:
    1. app.state.alert_engine (set in lifespan) - best for single process
    2. Create new instance (fallback, loses state between requests)
    """
    if request is not None:
        engine = getattr(request.app.state, "alert_engine", None)
        if isinstance(engine, AlertEngine):
            return engine

    # Fallback (not ideal for circuit breaker state across requests)
    return AlertEngine()


@router.get("/status", summary="Get current webhook and circuit breaker status")
def get_alerting_status(request: Request) -> dict[str, Any]:
    """Return current alerting/webhook health.

    Uses the shared AlertEngine from app.state when available.
    """
    engine = _get_engine(request)
    status = engine.get_webhook_status()

    return {
        "ok": True,
        "webhook": status,
        "source": "app.state" if hasattr(request.app.state, "alert_engine") else "new_instance",
        "note": (
            "Circuit breaker state is per-process. "
            "For multi-worker consistency, use Redis or central store (TASK-07)."
        ),
    }


@router.post("/reset-circuit-breaker", summary="Reset the webhook circuit breaker")
def reset_circuit_breaker(request: Request) -> dict[str, Any]:
    """Manually reset the circuit breaker (ops / testing use)."""
    engine = _get_engine(request)
    engine.reset_circuit_breaker()

    return {
        "ok": True,
        "message": "Circuit breaker has been reset.",
        "new_status": engine.get_webhook_status(),
    }