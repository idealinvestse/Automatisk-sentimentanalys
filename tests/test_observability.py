"""Tests for unified observability (logging, status reporter, error helpers)."""

from __future__ import annotations

import json
import logging
from io import StringIO

import pytest

from src.analysis.registry import _analyzer_error_result, ensure_analyzers_loaded, run_analyzers
from src.core.logging_config import (
    JSONFormatter,
    get_logger,
    log_context,
    request_id_var,
    resolve_log_level,
)
from src.core.models import AnalysisContext, Segment
from src.core.status import StatusReporter, reset_status_reporter


@pytest.fixture(autouse=True)
def _reset_status():
    reset_status_reporter()
    yield
    reset_status_reporter()


def test_resolve_log_level_from_sentiment_log_level(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.setenv("SENTIMENT_LOG_LEVEL", "WARNING")
    monkeypatch.delenv("SENTIMENT_DEV", raising=False)
    assert resolve_log_level() == logging.WARNING


def test_resolve_log_level_dev_defaults_debug(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("SENTIMENT_LOG_LEVEL", raising=False)
    monkeypatch.setenv("SENTIMENT_DEV", "1")
    assert resolve_log_level() == logging.DEBUG


def test_json_formatter_includes_request_id():
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JSONFormatter())
    root = logging.getLogger("test_obs_json")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    token = request_id_var.set("req-123")
    try:
        root.info("hello")
    finally:
        request_id_var.reset(token)
    payload = json.loads(stream.getvalue().strip())
    assert payload["request_id"] == "req-123"
    assert payload["message"] == "hello"


def test_log_context_injects_fields():
    with log_context(component="pipeline", phase="transcribe", job_id="job-1"):
        adapter = get_logger("test_obs_ctx")
        assert adapter.extra["component"] == "pipeline"
        assert adapter.extra["phase"] == "transcribe"
        assert adapter.extra["job_id"] == "job-1"


def test_status_reporter_notifies_listeners(tmp_path):
    reporter = StatusReporter(events_path=tmp_path / "events.jsonl", file_enabled=True)
    seen: list[str] = []

    def listener(event):
        seen.append(event.message)

    reporter.add_listener(listener)
    reporter.phase("pipeline", "start", "Test start")
    assert seen == ["Test start"]
    assert reporter.recent_events(limit=1)[0]["component"] == "pipeline"


def test_status_reporter_writes_jsonl(tmp_path):
    path = tmp_path / "process_events.jsonl"
    reporter = StatusReporter(events_path=path, file_enabled=True)
    reporter.info("cli", "run", "Script started")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["level"] == "INFO"
    assert payload["message"] == "Script started"


def test_analyzer_error_result_shape():
    result = _analyzer_error_result("sentiment", ValueError("boom"))
    assert "sentiment" in result
    assert result["sentiment"]["error_code"] == "analysis_failed"
    assert result["sentiment"]["fallback"] is True


def test_run_analyzers_stores_error_on_failure():
    ensure_analyzers_loaded()

    class BrokenAnalyzer:
        @property
        def name(self) -> str:
            return "broken_test_analyzer"

        @property
        def requires(self) -> list[str]:
            return []

        def analyze(self, ctx: AnalysisContext) -> dict:
            raise RuntimeError("analyzer exploded")

    from src.analysis import registry as reg

    reg.register_analyzer_class("broken_test_analyzer", BrokenAnalyzer)
    ctx = AnalysisContext(
        segments=[Segment(start=0.0, end=1.0, text="hej")],
    )
    results = run_analyzers(ctx, selected=["broken_test_analyzer"])
    assert "broken_test_analyzer" in results
    assert results["broken_test_analyzer"]["error_code"] == "analysis_failed"


def test_status_processes_endpoint():
    from fastapi.testclient import TestClient

    from src.api.app import create_app

    app = create_app()
    client = TestClient(app)
    response = client.get("/status/processes?limit=5")
    assert response.status_code == 200
    body = response.json()
    assert "events" in body
    assert isinstance(body["events"], list)


def test_health_detail_endpoint():
    from fastapi.testclient import TestClient

    from src.api.app import create_app

    app = create_app()
    client = TestClient(app)
    response = client.get("/status/health/detail")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "analyzers" in body
    assert "asr" in body
