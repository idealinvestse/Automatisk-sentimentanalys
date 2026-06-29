"""Tests for optional OpenTelemetry tracing helpers."""

from __future__ import annotations

import importlib

import pytest

import src.core.tracing as tracing_module


@pytest.fixture(autouse=True)
def _reset_tracing_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts with a clean tracing module state."""
    monkeypatch.delenv("OTEL_ENABLED", raising=False)
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    importlib.reload(tracing_module)
    yield
    importlib.reload(tracing_module)


def test_init_tracing_disabled_by_default() -> None:
    tracing_module.init_tracing()
    with tracing_module.span("noop") as current:
        assert current is None


def test_init_tracing_enabled_sets_tracer(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("opentelemetry")
    monkeypatch.setenv("OTEL_ENABLED", "true")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "test-service")
    importlib.reload(tracing_module)

    tracing_module.init_tracing()
    with tracing_module.span("test-span", job_id="j1") as current:
        assert current is not None


def test_init_tracing_skips_when_already_initialized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_ENABLED", "false")
    importlib.reload(tracing_module)
    tracing_module._initialized = True
    tracing_module.init_tracing("should-not-reinit")
    assert tracing_module._tracer is None


def test_span_noop_when_tracer_uninitialized() -> None:
    with tracing_module.span("x", phase="test") as current:
        assert current is None
