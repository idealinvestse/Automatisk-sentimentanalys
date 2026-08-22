"""Tests for transcription job registry and API endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.api import app
from src.api.app import create_app
from src.api.settings import get_api_settings
from src.api.transcription_job_store import RedisJobStore, SqliteJobStore, create_job_store
from src.api.transcription_jobs import TranscriptionJob, TranscriptionJobRegistry, get_job_registry


def test_job_survives_registry_recreate(tmp_path) -> None:
    db = tmp_path / "jobs.db"
    store = SqliteJobStore(db)
    reg1 = TranscriptionJobRegistry(store=store)
    reg1.register("j1", "transcribe")
    reg2 = TranscriptionJobRegistry(store=SqliteJobStore(db))
    assert reg2.get("j1") is not None
    assert reg2.get("j1").status == "running"


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


def test_transcribe_http_409_when_job_cancelled_during_asr(monkeypatch, tmp_path) -> None:
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"RIFF....WAVE")
    monkeypatch.setenv("API_MEDIA_ROOT", str(tmp_path))
    get_api_settings.cache_clear()
    client = TestClient(create_app(), raise_server_exceptions=False)
    job_id = "cancel-during-asr"

    def _cancel_then_ok(**_kwargs):
        get_job_registry(client.app).cancel(job_id)
        return {"text": "hej", "segments": [{"text": "hej", "start": 0, "end": 1}]}

    with patch("src.api.routers.transcription.transcribe_helper", side_effect=_cancel_then_ok):
        response = client.post(
            "/transcribe",
            json={"audio_path": str(wav)},
            headers={"X-Transcription-Job-Id": job_id},
        )
    assert response.status_code == 409
    assert response.json()["error_code"] == "conflict"
    assert "request_id" in response.json()


def test_transcribe_run_partial_analysis_http(monkeypatch, tmp_path) -> None:
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"RIFF....WAVE")
    monkeypatch.setenv("API_MEDIA_ROOT", str(tmp_path))
    get_api_settings.cache_clear()
    client = TestClient(create_app(), raise_server_exceptions=False)
    fake_report = MagicMock()
    fake_report.sentiment_results = [{"label": "neutral", "score": 0.5}]
    fake_report.results = {
        "partial": {"incremental": True},
        "analyzer_routing": {},
        "degradation": {},
    }
    with (
        patch(
            "src.api.routers.transcription.transcribe_helper",
            return_value={"text": "hej", "segments": [{"text": "hej", "start": 0, "end": 1}]},
        ),
        patch("src.api.dependencies.create_pipeline") as mock_pipe,
    ):
        mock_pipe.return_value.analyze_segments_partial.return_value = fake_report
        response = client.post(
            "/transcribe",
            json={"audio_path": str(wav), "run_partial_analysis": True},
        )
    assert response.status_code == 200
    assert response.json()["partial_analysis"]["sentiment_count"] == 1
    assert response.json()["partial_analysis"]["partial"]["incremental"] is True


class _DictRedis:
    def __init__(self) -> None:
        self.kv: dict[str, str] = {}
        self.z: dict[str, dict[str, float]] = {}

    def ping(self) -> bool:
        return True

    def set(self, key: str, value: str) -> None:
        self.kv[key] = value

    def get(self, key: str) -> str | None:
        return self.kv.get(key)

    def delete(self, key: str) -> None:
        self.kv.pop(key, None)

    def zadd(self, key: str, mapping: dict[str, float]) -> None:
        self.z.setdefault(key, {}).update(mapping)

    def zrevrange(self, key: str, start: int, end: int) -> list[str]:
        items = sorted(self.z.get(key, {}).items(), key=lambda item: -item[1])
        if end < 0:
            return [name for name, _ in items[start:]]
        return [name for name, _ in items[start : end + 1]]

    def zrem(self, key: str, member: str) -> None:
        self.z.get(key, {}).pop(member, None)


def test_redis_job_store_roundtrip() -> None:
    store = RedisJobStore(_DictRedis())
    job = TranscriptionJob(job_id="rj1", kind="transcribe")
    store.upsert(job)
    assert store.get("rj1")["job_id"] == "rj1"
    assert store.get("missing") is None
    assert store.list(limit=5)[0]["job_id"] == "rj1"
    store.complete("rj1", status="completed")
    assert store.get("rj1")["status"] == "completed"
    store.cancel("rj1")
    assert store.get("rj1")["cancelled"] is True
    store.delete("rj1")
    assert store.get("rj1") is None
    store.complete("missing")
    store.cancel("missing")


def test_create_job_store_prefers_redis_then_sqlite(tmp_path) -> None:
    redis_store = create_job_store(state_dir=tmp_path, use_redis=True, redis_client=_DictRedis())
    assert isinstance(redis_store, RedisJobStore)

    class _Dead:
        def ping(self) -> bool:
            raise ConnectionError("down")

    sqlite_store = create_job_store(state_dir=tmp_path, use_redis=True, redis_client=_Dead())
    assert isinstance(sqlite_store, SqliteJobStore)
    sqlite_store.delete("ghost")
