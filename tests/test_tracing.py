"""Tests for optional OpenTelemetry tracing helpers."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest

import src.core.tracing as tracing_module


@pytest.fixture(autouse=True)
def _reset_tracing_state() -> None:
    tracing_module._tracer = None
    tracing_module._initialized = False
    yield
    tracing_module._tracer = None
    tracing_module._initialized = False


class TestInitTracing:
    def test_disabled_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OTEL_ENABLED", raising=False)
        tracing_module.init_tracing("test-service")
        assert tracing_module._tracer is None
        assert tracing_module._initialized is True

    def test_init_only_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OTEL_ENABLED", "false")
        tracing_module.init_tracing("first")
        tracing_module._tracer = MagicMock(name="kept-tracer")
        tracing_module.init_tracing("second")
        assert tracing_module._tracer is not None

    def test_enabled_with_mocked_otel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OTEL_ENABLED", "true")
        monkeypatch.setenv("OTEL_SERVICE_NAME", "sentiment-test")

        mock_trace = MagicMock()
        mock_tracer = MagicMock()
        mock_trace.get_tracer.return_value = mock_tracer

        with patch.dict(
            "sys.modules",
            {
                "opentelemetry": MagicMock(trace=mock_trace),
                "opentelemetry.sdk.resources": MagicMock(
                    Resource=MagicMock(create=MagicMock(return_value=MagicMock()))
                ),
                "opentelemetry.sdk.trace": MagicMock(
                    TracerProvider=MagicMock(),
                ),
                "opentelemetry.sdk.trace.export": MagicMock(
                    BatchSpanProcessor=MagicMock(),
                    ConsoleSpanExporter=MagicMock(),
                ),
            },
        ):
            importlib.reload(tracing_module)
            tracing_module.init_tracing()
            assert tracing_module._tracer is mock_tracer

    def test_import_error_is_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OTEL_ENABLED", "true")
        with patch.dict("sys.modules", {"opentelemetry": None}):
            importlib.reload(tracing_module)
            tracing_module.init_tracing("svc")
            assert tracing_module._tracer is None


class TestSpan:
    def test_span_noop_when_tracer_uninitialized(self) -> None:
        with tracing_module.span("test-span", foo="bar") as current:
            assert current is None

    def test_span_sets_attributes(self) -> None:
        mock_span = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_span)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_cm
        tracing_module._tracer = mock_tracer

        with tracing_module.span("analyze", request_id="abc", skipped=None) as current:
            assert current is mock_span

        mock_tracer.start_as_current_span.assert_called_once_with("analyze")
        mock_span.set_attribute.assert_called_once_with("request_id", "abc")
