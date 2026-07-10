"""Tests for transcription job registry and API endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api import app
from src.api.app import create_app
from src.api.settings import get_api_settings
from src.api.transcription_jobs import TranscriptionJobRegistry, get_job_registry


def test_job_registry_cancel() -> None:
    registry = TranscriptionJobRegistry()
    job = registry.register("job-1", "batch_transcribe", total=5)
    assert job.status == "running"
    assert registry.cancel("job-1") == "cancelled"
    assert registry.is_cancelled("job-1") is True
    assert registry.cancel("job-1") == "already_cancelled"
    assert registry.cancel("missing") == "not_found"


def test_job_registry_cancel_does_not_overwrite_terminal_status() -> None:
    registry = TranscriptionJobRegistry()
    registry.register("done-job", "transcribe")
    registry.complete("done-job", status="completed")
    assert registry.cancel("done-job") == "already_finished"
    assert registry.get("done-job") is not None
    assert registry.get("done-job").status == "completed"

    registry.register("fail-job", "transcribe")
    registry.complete("fail-job", status="failed")
    assert registry.cancel("fail-job") == "already_finished"
    assert registry.get("fail-job").status == "failed"


def test_job_registry_get_complete_update_meta() -> None:
    registry = TranscriptionJobRegistry()
    registry.register("meta-job", "transcribe")
    registry.update_meta("meta-job", progress=0.5)
    job = registry.get("meta-job")
    assert job is not None
    assert job.meta["progress"] == 0.5
    registry.complete("meta-job")
    assert registry.get("meta-job").status == "completed"
    assert registry.get("missing") is None


def test_job_registry_max_jobs_evicts_terminal_only() -> None:
    registry = TranscriptionJobRegistry(max_jobs=3)
    registry.register("r1", "transcribe")
    registry.register("r2", "transcribe")
    registry.complete("r1")
    registry.complete("r2")
    registry.register("r3", "transcribe")
    registry.register("r4", "transcribe")  # should evict oldest terminal
    assert registry.get("r1") is None
    assert registry.get("r3") is not None
    assert registry.get("r4") is not None
    assert registry.get("r2") is not None or registry.get("r3") is not None


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


def test_get_job_endpoint_200_and_404() -> None:
    get_api_settings.cache_clear()
    client = TestClient(create_app())
    registry = get_job_registry(client.app)
    registry.register("get-me", "transcribe")

    ok = client.get("/transcription/jobs/get-me")
    assert ok.status_code == 200
    assert ok.json()["job_id"] == "get-me"
    assert ok.json()["status"] == "running"

    missing = client.get("/transcription/jobs/does-not-exist")
    assert missing.status_code == 404
    assert "not found" in missing.json()["detail"].lower()


def test_cancel_endpoint() -> None:
    get_api_settings.cache_clear()
    client = TestClient(create_app())
    registry = get_job_registry(client.app)
    registry.register("cancel-me", "scan_process")
    response = client.post("/transcription/jobs/cancel-me/cancel")
    assert response.status_code == 200
    assert response.json()["cancelled"] is True
    assert registry.is_cancelled("cancel-me") is True


def test_cancel_endpoint_idempotent_and_finished() -> None:
    get_api_settings.cache_clear()
    client = TestClient(create_app())
    registry = get_job_registry(client.app)

    registry.register("again", "transcribe")
    assert client.post("/transcription/jobs/again/cancel").status_code == 200
    assert client.post("/transcription/jobs/again/cancel").status_code == 200

    registry.register("finished", "transcribe")
    registry.complete("finished", status="completed")
    conflict = client.post("/transcription/jobs/finished/cancel")
    assert conflict.status_code == 409
    assert "finished" in conflict.json()["detail"].lower()
    assert registry.get("finished").status == "completed"

    missing = client.post("/transcription/jobs/nope/cancel")
    assert missing.status_code == 404
