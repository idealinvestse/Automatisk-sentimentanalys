from __future__ import annotations

import time
from pathlib import Path

from src.api.services.analysis_jobs import AnalysisJobManager


def _wait(manager: AnalysisJobManager, job_id: str, timeout: float = 2.0) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = manager.get(job_id)
        if job and job.status not in {"queued", "running", "cancel_requested"}:
            return job.status
        time.sleep(0.01)
    raise AssertionError("analysis job did not finish")


def test_analysis_job_persists_status_and_result(tmp_path: Path) -> None:
    manager = AnalysisJobManager(tmp_path)
    job = manager.submit(
        {"segments": [{"text": "redigerad"}]},
        lambda payload: {"ok": True, "degraded": [], "count": len(payload["segments"])},
    )
    assert _wait(manager, job.job_id) == "completed"
    assert manager.result(job.job_id) == {"ok": True, "degraded": [], "count": 1}
    assert not (tmp_path / "analysis_jobs" / f"{job.job_id}.input.json").exists()
    manager.shutdown()


def test_analysis_job_idempotency_returns_existing_job(tmp_path: Path) -> None:
    manager = AnalysisJobManager(tmp_path)
    first = manager.submit(
        {"segments": [{"text": "redigerad"}]},
        lambda payload: {"degraded": []},
        idempotency_key="same-request",
    )
    second = manager.submit(
        {"segments": [{"text": "annan"}]},
        lambda payload: {"degraded": []},
        idempotency_key="same-request",
    )
    assert second.job_id == first.job_id
    _wait(manager, first.job_id)
    manager.shutdown()


def test_running_job_becomes_interrupted_after_restart(tmp_path: Path) -> None:
    root = tmp_path / "analysis_jobs"
    root.mkdir()
    (root / "abc.status.json").write_text(
        '{"job_id":"abc","status":"running","created_at":"2026-01-01T00:00:00+00:00",'
        '"updated_at":"2026-01-01T00:00:00+00:00","phase":"llm_holistic",'
        '"error_code":null,"result_available":false,"cancel_requested":false,"meta":{}}',
        encoding="utf-8",
    )
    manager = AnalysisJobManager(tmp_path)
    job = manager.get("abc")
    assert job is not None
    assert job.status == "interrupted"
    assert job.error_code == "backend_restarted"
    manager.shutdown()


def test_analysis_job_queue_limit_rejects_overflow(tmp_path: Path) -> None:
    manager = AnalysisJobManager(tmp_path, max_queued=2)
    slow_runner = lambda payload: {"degraded": []}  # noqa: E731
    manager.submit({"segments": [{"text": "a"}]}, slow_runner)
    manager.submit({"segments": [{"text": "b"}]}, slow_runner)
    import pytest

    with pytest.raises(OverflowError):
        manager.submit({"segments": [{"text": "c"}]}, slow_runner)
    manager.shutdown()
