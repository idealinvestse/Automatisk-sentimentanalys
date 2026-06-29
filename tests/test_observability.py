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
from src.core.metrics import STATUS_EVENTS_TOTAL, record_status_event
from src.core.models import AnalysisContext, Segment
from src.core.observability import (
    SamplingFilter,
    degrading_phase,
    job_scope,
    phase_timer,
    with_error_handling,
)
from src.core.status import StatusReporter, derive_job_status, reset_status_reporter


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


def test_get_logger_live_context_after_import():
    module_logger = get_logger("test_live_ctx_module")
    with log_context(component="pipeline", phase="transcribe", job_id="job-live"):
        _msg, kwargs = module_logger.process("hello", {})
        assert kwargs["extra"]["job_id"] == "job-live"
        assert kwargs["extra"]["component"] == "pipeline"


def test_log_context_injects_fields():
    adapter = get_logger("test_obs_ctx")
    with log_context(component="pipeline", phase="transcribe", job_id="job-1"):
        _msg, kwargs = adapter.process("hello", {})
        assert kwargs["extra"]["component"] == "pipeline"
        assert kwargs["extra"]["phase"] == "transcribe"
        assert kwargs["extra"]["job_id"] == "job-1"


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
    assert path.is_file()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["level"] == "INFO"
    assert payload["message"] == "Script started"


def test_status_error_includes_error_code_and_exception_type(tmp_path):
    reporter = StatusReporter(events_path=tmp_path / "events.jsonl", file_enabled=False)
    try:
        raise ValueError("boom")
    except ValueError as exc:
        reporter.error("pipeline", "test", "failed", exc=exc, error_code="analysis_failed")
    event = reporter.recent_events(limit=1)[0]
    assert event["error_code"] == "analysis_failed"
    assert event["exception_type"] == "ValueError"


def test_status_dedup_collapses_repeated_events(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTIMENT_STATUS_DEDUP_WINDOW_S", "5")
    reporter = StatusReporter(events_path=tmp_path / "events.jsonl", file_enabled=False)
    reporter.info("pipeline", "progress", "same message")
    reporter.info("pipeline", "progress", "same message")
    reporter.info("pipeline", "progress", "same message")
    reporter.phase("pipeline", "next", "different")
    messages = [e["message"] for e in reporter.recent_events(limit=10)]
    assert any("upprepad 3" in m for m in messages)


def test_sampling_filter_drops_debug_records():
    filt = SamplingFilter({"chatty": 3})
    record = logging.LogRecord("chatty.module", logging.DEBUG, "", 0, "msg", (), None)
    results = [filt.filter(record) for _ in range(6)]
    assert results.count(True) == 2


def test_phase_timer_emits_start_and_complete(tmp_path, monkeypatch):
    reporter = StatusReporter(events_path=tmp_path / "events.jsonl", file_enabled=False)
    monkeypatch.setattr("src.core.observability.get_status_reporter", lambda: reporter)
    with phase_timer("pipeline", "test_phase"):
        pass
    events = reporter.recent_events(limit=10)
    phases = [e["phase"] for e in events]
    assert "test_phase" in phases
    assert any("start test_phase" in e["message"] for e in events)
    assert any("klar test_phase" in e["message"] for e in events)


def test_phase_timer_emits_error_on_failure(tmp_path, monkeypatch):
    reporter = StatusReporter(events_path=tmp_path / "events.jsonl", file_enabled=False)
    monkeypatch.setattr("src.core.observability.get_status_reporter", lambda: reporter)
    with pytest.raises(RuntimeError), phase_timer("pipeline", "fail_phase"):
        raise RuntimeError("kaboom")
    events = reporter.recent_events(limit=10)
    assert any(e["level"] == "ERROR" for e in events)


def test_with_error_handling_degrades(tmp_path):
    StatusReporter(events_path=tmp_path / "events.jsonl", file_enabled=False)

    @with_error_handling("pipeline", "step_x", result_key="step_x")
    def broken() -> dict:
        raise RuntimeError("nope")

    result = broken()
    assert "step_x" in result
    assert result["step_x"]["fallback"] is True


def test_job_scope_binds_job_id(tmp_path):
    reporter = StatusReporter(events_path=tmp_path / "events.jsonl", file_enabled=False)
    with job_scope("job-abc", component="api"):
        reporter.info("api", "request", "handling")
    event = reporter.recent_events(limit=1)[0]
    assert event["job_id"] == "job-abc"


def test_degrading_phase_writes_result_on_failure(tmp_path):
    StatusReporter(events_path=tmp_path / "events.jsonl", file_enabled=False)
    results: dict = {}
    with degrading_phase(
        "pipeline", "agent_performance", results=results, result_key="agent_performance"
    ):
        raise RuntimeError("perf failed")
    assert "agent_performance" in results
    assert results["agent_performance"]["fallback"] is True


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


def test_record_status_event_increments_counter():
    if STATUS_EVENTS_TOTAL is None:
        pytest.skip("prometheus_client not installed")
    before = STATUS_EVENTS_TOTAL.labels(
        level="INFO", component="test", error_code="none"
    )._value.get()
    record_status_event("INFO", "test", "")
    after = STATUS_EVENTS_TOTAL.labels(
        level="INFO", component="test", error_code="none"
    )._value.get()
    assert after == before + 1


def test_derive_job_status():
    events = [
        {
            "job_id": "j1",
            "phase": "transcribe",
            "component": "pipeline",
            "level": "PHASE",
            "ts": "2026-01-01T00:00:00",
        },
        {
            "job_id": "j1",
            "phase": "complete",
            "component": "pipeline",
            "level": "PHASE",
            "ts": "2026-01-01T00:01:00",
            "progress": 1.0,
        },
    ]
    summary = derive_job_status(events, "j1")
    assert summary["found"] is True
    assert summary["current_phase"] == "complete"


def test_status_processes_endpoint():
    from fastapi.testclient import TestClient

    from src.api.app import create_app

    app = create_app()
    client = TestClient(app)
    response = client.get("/status/processes?limit=5&component=pipeline")
    assert response.status_code == 200
    body = response.json()
    assert "events" in body


def test_job_status_endpoint():
    from fastapi.testclient import TestClient

    from src.api.app import create_app
    from src.core.status import get_status_reporter

    get_status_reporter().phase("pipeline", "transcribe", "start", job_id="job-xyz")
    app = create_app()
    client = TestClient(app)
    response = client.get("/status/jobs/job-xyz")
    assert response.status_code == 200
    body = response.json()
    assert body["job_id"] == "job-xyz"
    assert body["found"] is True


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


def test_jsonl_rotation_uses_rotating_handler(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTIMENT_STATUS_FILE_MAX_BYTES", "200")
    path = tmp_path / "process_events.jsonl"
    reporter = StatusReporter(events_path=path, file_enabled=True)
    for i in range(30):
        reporter.info("cli", "run", f"event number {i} with padding")
    assert path.is_file()
    backup = path.with_suffix(path.suffix + ".1")
    assert path.stat().st_size <= 250 or backup.exists()
