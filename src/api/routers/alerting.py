"""Alerting status and management endpoints.

Exposes webhook/circuit breaker health for dashboard and ops.
"""

from __future__ import annotations

from fastapi import APIRouter

from src.alerting import AlertEngine, load_alerting_config

router = APIRouter(prefix="/alerting", tags=["alerting"])

# Shared engine instance for status tracking (simple approach for v1)
_status_engine: AlertEngine | None = None


def get_status_engine() -> AlertEngine:
    """Get or create a shared AlertEngine for status queries."""
    global _status_engine
    if _status_engine is None:
        _status_engine = AlertEngine()
    return _status_engine


@router.get("/status", summary="Get current webhook and circuit breaker status")
def get_alerting_status() -> dict:
    """Return current alerting/webhook health.

    Includes circuit breaker state so the dashboard can show real-time status.
    """
    engine = get_status_engine()
    status = engine.get_webhook_status()
    return {
        "ok": True,
        "webhook": status,
        "note": "Circuit breaker state is tracked per AlertEngine instance. "
        "For full shared state across requests, consider a singleton or Redis-backed store.",
    }


@router.post("/reset-circuit-breaker", summary="Reset the webhook circuit breaker")
def reset_circuit_breaker() -> dict:
    """Manually reset the circuit breaker (ops / testing use)."""
    engine = get_status_engine()
    engine.reset_circuit_breaker()
    return {
        "ok": True,
        "message": "Circuit breaker has been reset.",
        "new_status": engine.get_webhook_status(),
    }