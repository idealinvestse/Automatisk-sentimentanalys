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


def test_server_shim_imports() -> None:
    from src.api.server import app as server_app

    assert server_app.title == "Swedish Sentiment API"


def test_health_ok() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") in ("ok", "healthy")


def test_metrics_endpoint() -> None:
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    assert "alerting_circuit_breaker_open" in body or "prometheus_client not installed" in body


def test_http_request_metrics_recorded() -> None:
    r = client.get("/health")
    assert r.status_code == 200
    metrics = client.get("/metrics").text
    if "prometheus_client not installed" in metrics:
        pytest.skip("prometheus_client not installed")
    assert "http_requests_total" in metrics
    assert 'method="GET"' in metrics


def test_openapi_has_core_paths() -> None:
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json().get("paths", {})
    for path in ("/health", "/analyze", "/analyze_pipeline", "/analyze_pipeline/partial"):
        assert path in paths


def test_analyze_pipeline_partial_happy_mocked() -> None:
    fake_report = MagicMock()
    fake_report.sentiment_results = [{"label": "neutral", "score": 0.5}]
    fake_report.intent_results = []
    fake_report.summary = {}
    fake_report.topics = {}
    fake_report.insights = {}
    fake_report.risks = {}
    fake_report.processing_time_s = 0.05
    fake_report.llm = {}
    fake_report.results = {"partial": {"incremental": True, "reconciled": False}}

    with patch("src.api.dependencies.CallAnalysisPipeline") as mock_pipe:
        inst = mock_pipe.return_value
        inst.analyze_segments_partial.return_value = fake_report
        r = client.post(
            "/analyze_pipeline/partial",
            json={"segments": [{"text": "Hej", "start": 0, "end": 1}]},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["results"]["partial"]["incremental"] is True


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


def test_request_id_preserved_in_error_body() -> None:
    r = client.post(
        "/analyze",
        json={"texts": []},
        headers={"X-Request-ID": "client-provided-id"},
    )
    assert r.status_code == 422
    assert r.headers["X-Request-ID"] == "client-provided-id"
    assert r.json()["request_id"] == "client-provided-id"


def test_production_guard_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_REQUIRE_AUTH", "true")
    monkeypatch.delenv("SENTIMENT_API_KEY", raising=False)
    get_api_settings.cache_clear()
    from src.api.settings import validate_production_settings
    from src.core.errors import ConfigurationError

    settings = get_api_settings()
    with pytest.raises(ConfigurationError, match="SENTIMENT_API_KEY"):
        validate_production_settings(settings)


def test_production_guard_requires_media_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_REQUIRE_MEDIA_ROOT", "true")
    monkeypatch.delenv("API_MEDIA_ROOT", raising=False)
    get_api_settings.cache_clear()
    from src.api.settings import validate_production_settings
    from src.core.errors import ConfigurationError

    settings = get_api_settings()
    with pytest.raises(ConfigurationError, match="API_MEDIA_ROOT"):
        validate_production_settings(settings)


def test_api_production_implies_auth_and_media_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_PRODUCTION", "true")
    monkeypatch.delenv("API_REQUIRE_AUTH", raising=False)
    monkeypatch.delenv("API_REQUIRE_MEDIA_ROOT", raising=False)
    monkeypatch.delenv("SENTIMENT_API_KEY", raising=False)
    monkeypatch.delenv("API_MEDIA_ROOT", raising=False)
    get_api_settings.cache_clear()
    from src.api.settings import validate_production_settings
    from src.core.errors import ConfigurationError

    settings = get_api_settings()
    with pytest.raises(ConfigurationError, match="SENTIMENT_API_KEY"):
        validate_production_settings(settings)

    monkeypatch.setenv("SENTIMENT_API_KEY", "test-key")
    get_api_settings.cache_clear()
    settings = get_api_settings()
    with pytest.raises(ConfigurationError, match="API_MEDIA_ROOT"):
        validate_production_settings(settings)


# --- Edge AI router tests ---------------------------------------------------


def test_edge_analyze_text_happy_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.edge.contracts import EdgeAnalysisResult, EdgeSegmentResult

    def fake_analyze_text(text: str, *, profile: str = "callcenter") -> EdgeAnalysisResult:
        return EdgeAnalysisResult(
            profile=profile,
            segments=[
                EdgeSegmentResult(
                    text=text,
                    sentiment_label="positiv",
                    sentiment_score=0.9,
                    intent="information_request",
                )
            ],
            summary="Offline analysis (callcenter)",
        )

    monkeypatch.setattr("src.api.routers.edge.analyze_text_offline", fake_analyze_text)
    r = client.post(
        "/edge/analyze-text", json={"text": "Tack för hjälpen!", "profile": "callcenter"}
    )
    assert r.status_code == 200
    data = r.json()
    assert data["profile"] == "callcenter"
    assert data["offline"] is True
    assert data["llm_used"] is False
    assert len(data["segments"]) == 1
    assert data["segments"][0]["sentiment_label"] == "positiv"


def test_edge_analyze_text_empty_422() -> None:
    r = client.post("/edge/analyze-text", json={"text": "", "profile": "callcenter"})
    assert r.status_code in (400, 422)


def test_edge_analyze_segments_happy_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.edge.contracts import EdgeAnalysisResult, EdgeSegmentResult

    def fake_analyze_segments(segments, *, profile: str = "callcenter") -> EdgeAnalysisResult:
        return EdgeAnalysisResult(
            profile=profile,
            segments=[
                EdgeSegmentResult(text=s.get("text", ""), sentiment_label="neutral", intent="other")
                for s in segments
            ],
        )

    monkeypatch.setattr("src.api.routers.edge.analyze_segments_offline", fake_analyze_segments)
    r = client.post(
        "/edge/analyze-segments",
        json={
            "segments": [
                {"text": "Hej", "speaker": "Agent"},
                {"text": "Hej då", "speaker": "Kund"},
            ],
            "profile": "callcenter",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data["segments"]) == 2
    assert data["segments"][0]["text"] == "Hej"


def test_edge_analyze_segments_empty_422() -> None:
    r = client.post("/edge/analyze-segments", json={"segments": [], "profile": "callcenter"})
    assert r.status_code in (400, 422)


def test_edge_paths_in_openapi() -> None:
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json().get("paths", {})
    assert "/edge/analyze-text" in paths
    assert "/edge/analyze-segments" in paths
