"""In-memory registry for active transcription jobs (status + cancellation)."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class TranscriptionJob:
    job_id: str
    kind: str  # transcribe | batch_transcribe | scan_process
    status: str = "running"  # running | cancelled | completed | failed
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    cancel_event: threading.Event = field(default_factory=threading.Event)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "status": self.status,
            "created_at": self.created_at,
            "cancelled": self.cancel_event.is_set(),
            "meta": dict(self.meta),
        }


class TranscriptionJobRegistry:
    """Thread-safe job store keyed by X-Transcription-Job-Id."""

    def __init__(self, *, max_jobs: int = 100) -> None:
        self._jobs: dict[str, TranscriptionJob] = {}
        self._lock = threading.Lock()
        self._max_jobs = max_jobs

    def register(self, job_id: str, kind: str, **meta: Any) -> TranscriptionJob:
        job = TranscriptionJob(job_id=job_id, kind=kind, meta=dict(meta))
        with self._lock:
            self._jobs[job_id] = job
            if len(self._jobs) > self._max_jobs:
                oldest = sorted(self._jobs.values(), key=lambda j: j.created_at)
                for stale in oldest[: len(self._jobs) - self._max_jobs]:
                    if stale.status in ("completed", "failed", "cancelled"):
                        self._jobs.pop(stale.job_id, None)
        return job

    def get(self, job_id: str) -> TranscriptionJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, *, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [j.to_dict() for j in jobs[:limit]]

    def update_meta(self, job_id: str, **meta: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.meta.update(meta)

    def complete(self, job_id: str, *, status: str = "completed") -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = status

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            job.cancel_event.set()
            job.status = "cancelled"
            return True

    def is_cancelled(self, job_id: str | None) -> bool:
        if not job_id:
            return False
        with self._lock:
            job = self._jobs.get(job_id)
            return bool(job and job.cancel_event.is_set())


def get_job_registry(app: Any) -> TranscriptionJobRegistry:
    registry = getattr(app.state, "transcription_jobs", None)
    if registry is None:
        registry = TranscriptionJobRegistry()
        app.state.transcription_jobs = registry
    return registry
