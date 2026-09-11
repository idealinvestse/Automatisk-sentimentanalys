"""Bounded persistent background jobs for long post-transcription analysis."""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class AnalysisJob:
    """Persistent state for one long-running analysis."""

    job_id: str
    status: str
    created_at: str
    updated_at: str
    phase: str = "queued"
    error_code: str | None = None
    result_available: bool = False
    cancel_requested: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return public status fields without transcript content."""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "phase": self.phase,
            "error_code": self.error_code,
            "result_available": self.result_available,
            "cancel_requested": self.cancel_requested,
            "meta": dict(self.meta),
        }


class AnalysisJobManager:
    """Execute at most one local long-context analysis at a time."""

    def __init__(self, state_dir: str | Path, *, max_queued: int = 10) -> None:
        self.root = Path(state_dir) / "analysis_jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_queued = max_queued
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analysis-job")
        self._jobs: dict[str, AnalysisJob] = {}
        self._futures: dict[str, Future[Any]] = {}
        self._load()

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    def _status_path(self, job_id: str) -> Path:
        return self.root / f"{job_id}.status.json"

    def _input_path(self, job_id: str) -> Path:
        return self.root / f"{job_id}.input.json"

    def _result_path(self, job_id: str) -> Path:
        return self.root / f"{job_id}.result.json"

    @staticmethod
    def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)

    def _persist_status(self, job: AnalysisJob) -> None:
        self._atomic_write(self._status_path(job.job_id), job.to_dict())

    def _load(self) -> None:
        for path in self.root.glob("*.status.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("status") in {"queued", "running", "cancel_requested"}:
                    data["status"] = "interrupted"
                    data["phase"] = "interrupted"
                    data["updated_at"] = self._now()
                    data["error_code"] = "backend_restarted"
                job = AnalysisJob(**data)
                self._jobs[job.job_id] = job
                self._persist_status(job)
            except (OSError, json.JSONDecodeError, TypeError):
                continue

    def submit(
        self,
        payload: dict[str, Any],
        runner: Callable[[dict[str, Any]], dict[str, Any]],
        *,
        idempotency_key: str | None = None,
    ) -> AnalysisJob:
        """Persist redacted input and enqueue a bounded analysis job."""
        with self._lock:
            if idempotency_key:
                for existing in self._jobs.values():
                    if existing.meta.get("idempotency_key") == idempotency_key:
                        return existing
            queued = sum(job.status in {"queued", "running"} for job in self._jobs.values())
            if queued >= self.max_queued:
                raise OverflowError("analysis job queue is full")
            now = self._now()
            job = AnalysisJob(
                job_id=uuid.uuid4().hex,
                status="queued",
                created_at=now,
                updated_at=now,
                meta={"idempotency_key": idempotency_key} if idempotency_key else {},
            )
            self._jobs[job.job_id] = job
            self._atomic_write(self._input_path(job.job_id), payload)
            self._persist_status(job)
            self._futures[job.job_id] = self._executor.submit(self._run, job.job_id, runner)
            return job

    def _run(self, job_id: str, runner: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if job.cancel_requested:
                job.status = "cancelled"
                job.phase = "cancelled"
                job.updated_at = self._now()
                self._persist_status(job)
                self._forget_input(job_id)
                return
            job.status = "running"
            job.phase = "local_analysis"
            job.updated_at = self._now()
            self._persist_status(job)
        try:
            payload = json.loads(self._input_path(job_id).read_text(encoding="utf-8"))
            result = runner(payload)
            with self._lock:
                job = self._jobs[job_id]
                if job.cancel_requested:
                    job.status = "cancelled"
                    job.phase = "cancelled"
                else:
                    self._atomic_write(self._result_path(job_id), result)
                    job.result_available = True
                    job.status = "completed_degraded" if result.get("degraded") else "completed"
                    job.phase = "persisting"
                job.updated_at = self._now()
                self._persist_status(job)
            self._forget_input(job_id)
        except Exception as exc:
            with self._lock:
                job = self._jobs[job_id]
                job.status = "failed"
                job.phase = "failed"
                job.error_code = getattr(exc, "error_code", "analysis_job_failed")
                job.updated_at = self._now()
                self._persist_status(job)
            self._forget_input(job_id)

    def _forget_input(self, job_id: str) -> None:
        """Drop persisted transcript input after the job leaves the queue."""
        path = self._input_path(job_id)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    def get(self, job_id: str) -> AnalysisJob | None:
        """Return one job status."""
        with self._lock:
            return self._jobs.get(job_id)

    def result(self, job_id: str) -> dict[str, Any] | None:
        """Return a completed result when available."""
        path = self._result_path(job_id)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def cancel(self, job_id: str) -> str:
        """Request cooperative cancellation without releasing the active slot early."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return "not_found"
            if job.status in {"completed", "completed_degraded", "failed", "cancelled", "interrupted"}:
                return "already_finished"
            job.cancel_requested = True
            job.status = "cancel_requested" if job.status == "running" else "cancelled"
            job.phase = job.status
            job.updated_at = self._now()
            self._persist_status(job)
            return job.status

    def shutdown(self) -> None:
        """Stop accepting work while allowing the active request to finish."""
        self._executor.shutdown(wait=False, cancel_futures=True)
