"""Minimal API smoke tests — fast TestClient checks for a working API state."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.api.app import create_app
from src.api.settings import get_api_settings

client = TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _clear_api_settings_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent auth/env leakage between tests."""
    monkeypatch.delenv("SENTIMENT_API_KEY", raising=False)
    get_api_settings.cache_clear()


def test_app_imports() -> None:
    assert app.title == "Swedish Sentiment API"
    assert callable(create_app)


def test_health_ok() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") in ("ok", "healthy")


def test_openapi_has_core_paths() -> None:
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json().get("paths", {})
    for path in ("/health", "/analyze", "/analyze_pipeline"):
        assert path in paths


def test_analyze_happy_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_smart(texts, **kwargs):
        return ([{"label": "positiv", "score": 0.88}], {"profile": "default", "model": "fake"})

    monkeypatch.setattr("src.api.routers.text.analyze_smart", fake_smart)
    r = client.post("/analyze", json={"texts": ["Det här var fantastiskt!"]})
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
    assert data["meta"]["profile"] == "default"


def test_analyze_empty_texts_422() -> None:
    r = client.post("/analyze", json={"texts": []})
    assert r.status_code in (400, 422)


def test_analyze_pipeline_happy_mocked() -> None:
    fake_report = MagicMock()
    fake_report.sentiment_results = []
    fake_report.intent_results = []
    fake_report.summary = {}
    fake_report.topics = {}
    fake_report.insights = {}
    fake_report.risks = {}
    fake_report.processing_time_s = 0.12
    fake_report.llm = {}
    fake_report.results = {}

    with patch("src.api.dependencies.CallAnalysisPipeline") as mock_pipe:
        inst = mock_pipe.return_value
        inst.analyze_segments.return_value = fake_report
        r = client.post(
            "/analyze_pipeline",
            json={"segments": [{"text": "Hej", "start": 0, "end": 1}]},
        )
    assert r.status_code == 200
    data = r.json()
    assert "sentiment_results" in data
    assert "timestamp" in data


def test_app_state_initialized() -> None:
    assert hasattr(app.state, "cache")
    assert hasattr(app.state, "alert_engine")


def test_request_id_header() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert "X-Request-ID" in r.headers