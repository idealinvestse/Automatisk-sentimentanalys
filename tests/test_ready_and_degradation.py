"""Readiness probe and degradation helpers."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.degradation import collect_degraded_reasons
from src.api.settings import get_api_settings


def test_ready_endpoint_ok_in_dev(monkeypatch) -> None:
    monkeypatch.delenv("SENTIMENT_API_KEY", raising=False)
    monkeypatch.delenv("API_PRODUCTION", raising=False)
    monkeypatch.delenv("API_USE_REDIS_CACHE", raising=False)
    get_api_settings.cache_clear()
    client = TestClient(create_app())
    res = client.get("/ready")
    assert res.status_code == 200
    assert res.json()["status"] == "ready"


def test_collect_degraded_reasons_llm_skipped() -> None:
    report = SimpleNamespace(
        results={"degradation": {"reasons": ["diarization_unavailable"]}},
        llm={"llm_used": False, "skip_reason": "no_api_key"},
    )
    reasons = collect_degraded_reasons(report)
    assert "diarization_unavailable" in reasons
    assert any(r.startswith("llm:") for r in reasons)


def test_health_detail_includes_hub_backend(monkeypatch) -> None:
    monkeypatch.delenv("SENTIMENT_API_KEY", raising=False)
    get_api_settings.cache_clear()
    client = TestClient(create_app())
    res = client.get("/status/health/detail")
    assert res.status_code == 200
    body = res.json()
    assert "transcription_events" in body
    assert "backend" in body["transcription_events"]
    assert "checks" in body
    assert body["status"] in ("ok", "degraded")
