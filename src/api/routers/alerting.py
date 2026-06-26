"""Alerting status and management endpoints.

Exposes webhook/circuit breaker health for dashboard and ops.

Multi-worker note:
- Circuit breaker state is now taken from app.state.alert_engine (set in lifespan).
- This gives consistent state *within one worker*.
- For true cross-worker consistency (multiple uvicorn workers), a central store (Redis)
  or external coordination is still needed (future improvement).
"""