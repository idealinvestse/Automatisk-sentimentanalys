"""Optional OpenTelemetry tracing (PROD-01). Graceful no-op when OTEL not installed."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_tracer: Any = None
_initialized = False


def init_tracing(service_name: str | None = None) -> None:
    """Initialize OTEL tracer when ``OTEL_ENABLED=true``."""
    global _tracer, _initialized
    if _initialized:
        return
    _initialized = True
    if os.getenv("OTEL_ENABLED", "").lower() not in ("1", "true", "yes"):
        return
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    except ImportError:
        return

    name = service_name or os.getenv("OTEL_SERVICE_NAME", "sentiment-api")
    provider = TracerProvider(resource=Resource.create({"service.name": name}))
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(name)


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Create a span or no-op context."""
    if _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as current:
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(key, value)
        yield current
