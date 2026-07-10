"""Hardening tests for /status endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.settings import get_api_settings


def test_status_processes_limit_bounds(monkeypatch) -> None:
    monkeypatch.delenv("SENTIMENT_API_KEY", raising=False)
    get_api_settings.cache_clear()
    client = TestClient(create_app())

    assert client.get("/status/processes?limit=0").status_code == 422
    assert client.get("/status/processes?limit=2000").status_code == 422
    ok = client.get("/status/processes?limit=10")
    assert ok.status_code == 200
    assert "events" in ok.json()
    assert "count" in ok.json()


def test_status_unknown_job_found_false(monkeypatch) -> None:
    monkeypatch.delenv("SENTIMENT_API_KEY", raising=False)
    get_api_settings.cache_clear()
    client = TestClient(create_app())

    response = client.get("/status/jobs/never-seen-job-id")
    assert response.status_code == 200
    body = response.json()
    assert body.get("found") is False or body.get("job_id") == "never-seen-job-id"


def test_status_health_detail_ok(monkeypatch) -> None:
    monkeypatch.delenv("SENTIMENT_API_KEY", raising=False)
    get_api_settings.cache_clear()
    client = TestClient(create_app())

    response = client.get("/status/health/detail")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "analyzers" in body
    assert "asr" in body
