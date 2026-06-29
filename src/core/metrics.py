"""Shared Prometheus metrics for pipeline, analyzers, LLM, and cache (PROD-01)."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

try:
    from prometheus_client import Counter, Histogram
except ImportError:  # pragma: no cover
    Counter = None  # type: ignore[assignment,misc]
    Histogram = None  # type: ignore[assignment,misc]

PIPELINE_DURATION_SECONDS = (
    Histogram(
        "pipeline_duration_seconds",
        "End-to-end pipeline enrichment duration",
        ["endpoint", "profile", "has_llm"],
        buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
    )
    if Histogram is not None
    else None
)
ANALYZER_DURATION_SECONDS = (
    Histogram(
        "analyzer_duration_seconds",
        "Per-analyzer execution duration",
        ["analyzer"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    if Histogram is not None
    else None
)
LLM_REQUESTS_TOTAL = (
    Counter(
        "llm_requests_total",
        "LLM API requests",
        ["provider", "model", "outcome"],
    )
    if Counter is not None
    else None
)
LLM_REQUEST_DURATION_SECONDS = (
    Histogram(
        "llm_request_duration_seconds",
        "LLM request latency",
        ["provider", "model"],
        buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
    )
    if Histogram is not None
    else None
)
CACHE_OPERATIONS_TOTAL = (
    Counter(
        "cache_operations_total",
        "Aggregate cache operations",
        ["operation", "result"],
    )
    if Counter is not None
    else None
)
STATUS_EVENTS_TOTAL = (
    Counter(
        "sentiment_status_events_total",
        "Process status events emitted",
        ["level", "component", "error_code"],
    )
    if Counter is not None
    else None
)


def record_analyzer_duration(analyzer: str, duration_s: float) -> None:
    if ANALYZER_DURATION_SECONDS is not None:
        ANALYZER_DURATION_SECONDS.labels(analyzer=analyzer).observe(duration_s)


def record_pipeline_duration(
    endpoint: str, profile: str, has_llm: bool, duration_s: float
) -> None:
    if PIPELINE_DURATION_SECONDS is not None:
        PIPELINE_DURATION_SECONDS.labels(
            endpoint=endpoint,
            profile=profile,
            has_llm="true" if has_llm else "false",
        ).observe(duration_s)


def record_llm_request(
    provider: str, model: str, outcome: str, duration_s: float
) -> None:
    if LLM_REQUESTS_TOTAL is not None:
        LLM_REQUESTS_TOTAL.labels(provider=provider, model=model, outcome=outcome).inc()
    if LLM_REQUEST_DURATION_SECONDS is not None:
        LLM_REQUEST_DURATION_SECONDS.labels(provider=provider, model=model).observe(duration_s)


def record_cache_operation(operation: str, result: str) -> None:
    if CACHE_OPERATIONS_TOTAL is not None:
        CACHE_OPERATIONS_TOTAL.labels(operation=operation, result=result).inc()


def record_status_event(level: str, component: str, error_code: str = "") -> None:
    if STATUS_EVENTS_TOTAL is not None:
        STATUS_EVENTS_TOTAL.labels(
            level=level,
            component=component,
            error_code=error_code or "none",
        ).inc()


@contextmanager
def timed_analyzer(name: str) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        record_analyzer_duration(name, time.perf_counter() - started)


@contextmanager
def timed_pipeline(endpoint: str, profile: str, has_llm: bool) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        record_pipeline_duration(endpoint, profile, has_llm, time.perf_counter() - started)
