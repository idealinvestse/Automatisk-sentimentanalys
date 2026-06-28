"""Prometheus metrics for the Swedish Sentiment API (OBS-01)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..alerting_state import AlertingStateManager

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
except ImportError:  # pragma: no cover - optional until [api] extra installed
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    generate_latest = None  # type: ignore[assignment]
    Gauge = None  # type: ignore[assignment,misc]
    Counter = None  # type: ignore[assignment,misc]
    Histogram = None  # type: ignore[assignment,misc]

ALERTING_CIRCUIT_BREAKER_OPEN = (
    Gauge(
        "alerting_circuit_breaker_open",
        "Whether the alerting webhook circuit breaker is open (1) or closed (0)",
    )
    if Gauge is not None
    else None
)
ALERTING_CONSECUTIVE_FAILURES = (
    Gauge(
        "alerting_consecutive_failures",
        "Number of consecutive webhook delivery failures",
    )
    if Gauge is not None
    else None
)
APP_INFO = (
    Gauge(
        "sentiment_api_info",
        "Static API build info (always 1)",
        ["version"],
    )
    if Gauge is not None
    else None
)
HTTP_REQUESTS_TOTAL = (
    Counter(
        "http_requests_total",
        "Total HTTP requests",
        ["method", "path", "status"],
    )
    if Counter is not None
    else None
)
HTTP_REQUEST_DURATION_SECONDS = (
    Histogram(
        "http_request_duration_seconds",
        "HTTP request latency in seconds",
        ["method", "path"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    )
    if Histogram is not None
    else None
)


def init_app_info(version: str = "0.4.1") -> None:
    """Set static info gauge once at startup."""
    if APP_INFO is not None:
        APP_INFO.labels(version=version).set(1)


def update_alerting_metrics(state: AlertingStateManager) -> None:
    """Sync Prometheus gauges from AlertingStateManager."""
    if ALERTING_CIRCUIT_BREAKER_OPEN is None or ALERTING_CONSECUTIVE_FAILURES is None:
        return
    with state._lock:
        status = state._read()
    open_flag = 1.0 if status.get("circuit_breaker_open") else 0.0
    ALERTING_CIRCUIT_BREAKER_OPEN.set(open_flag)
    ALERTING_CONSECUTIVE_FAILURES.set(float(status.get("consecutive_failures", 0)))


def record_http_request(method: str, path: str, status: int, duration_s: float) -> None:
    """Record HTTP request count and latency (PROD-01)."""
    if HTTP_REQUESTS_TOTAL is not None:
        HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status=str(status)).inc()
    if HTTP_REQUEST_DURATION_SECONDS is not None:
        HTTP_REQUEST_DURATION_SECONDS.labels(method=method, path=path).observe(duration_s)


def render_metrics() -> tuple[bytes, str]:
    """Return Prometheus exposition payload and content type."""
    if generate_latest is None:
        return b"# prometheus_client not installed\n", CONTENT_TYPE_LATEST
    return generate_latest(), CONTENT_TYPE_LATEST
