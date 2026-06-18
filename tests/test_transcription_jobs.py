"""Tests for transcription job registry and API endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api import app
from src.api.settings import get_api_settings
from src.api.transcription_jobs import TranscriptionJobRegistry, get_job_registry


def test_job_registry_cancel() -> None:
    registry = TranscriptionJobRegistry()
    job = registry.register("job-1", "batch_transcribe", total=5)
    assert job.status == "running"
    assert registry.cancel("job-1") is True
    assert registry.is_cancelled("job-1") is True
    assert registry.cancel("missing") is False


def test_job_list_endpoint() -> None:
    get_api_settings.cache_clear()
    client = TestClient(app)
    registry = get_job_registry(app)
    registry.register("list-job", "transcribe")
    response = client.get("/transcription/jobs?limit=5")
    assert response.status_code == 200
    body = response.json()
    assert "jobs" in body
    assert any(j["job_id"] == "list-job" for j in body["jobs"])


def test_cancel_endpoint() -> None:
    get_api_settings.cache_clear()
    client = TestClient(app)
    registry = get_job_registry(app)
    registry.register("cancel-me", "scan_process")
    response = client.post("/transcription/jobs/cancel-me/cancel")
    assert response.status_code == 200
    assert response.json()["cancelled"] is True
    assert registry.is_cancelled("cancel-me") is True