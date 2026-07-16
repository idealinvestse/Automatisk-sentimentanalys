"""Registry for active transcription jobs (status + cancellation)."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .transcription_job_store import JobStore


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

    def __init__(
        self,
        *,
        max_jobs: int = 100,
        store: JobStore | None = None,
    ) -> None:
        self._jobs: dict[str, TranscriptionJob] = {}
        self._lock = threading.Lock()
        self._max_jobs = max_jobs
        self._store = store
        if store is not None:
            self._load_from_store()

    def _load_from_store(self) -> None:
        assert self._store is not None
        for record in self._store.list(limit=self._max_jobs * 2):
            job = self._job_from_record(record)
            self._jobs[job.job_id] = job

    @staticmethod
    def _job_from_record(record: dict[str, Any]) -> TranscriptionJob:
        cancel_event = threading.Event()
        cancelled = record.get("cancelled", False) or record.get("status") == "cancelled"
        if cancelled:
            cancel_event.set()
        return TranscriptionJob(
            job_id=record["job_id"],
            kind=record["kind"],
            status=record["status"],
            created_at=record["created_at"],
            cancel_event=cancel_event,
            meta=dict(record.get("meta") or {}),
        )

    def _persist(self, job: TranscriptionJob) -> None:
        if self._store is not None:
            self._store.upsert(job)

    def _delete_from_store(self, job_id: str) -> None:
        if self._store is not None and hasattr(self._store, "delete"):
            self._store.delete(job_id)  # type: ignore[attr-defined]

    def register(self, job_id: str, kind: str, **meta: Any) -> TranscriptionJob:
        job = TranscriptionJob(job_id=job_id, kind=kind, meta=dict(meta))
        with self._lock:
            self._jobs[job_id] = job
            self._persist(job)
            if len(self._jobs) > self._max_jobs:
                oldest = sorted(self._jobs.values(), key=lambda j: j.created_at)
                for stale in oldest[: len(self._jobs) - self._max_jobs]:
                    if stale.status in ("completed", "failed", "cancelled"):
                        self._jobs.pop(stale.job_id, None)
                        self._delete_from_store(stale.job_id)
        return job

    def get(self, job_id: str) -> TranscriptionJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                return job
        if self._store is not None:
            record = self._store.get(job_id)
            if record is None:
                return None
            job = self._job_from_record(record)
            with self._lock:
                self._jobs[job_id] = job
            return job
        return None

    def list_jobs(self, *, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [j.to_dict() for j in jobs[:limit]]

    def update_meta(self, job_id: str, **meta: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.meta.update(meta)
                self._persist(job)

    def complete(self, job_id: str, *, status: str = "completed") -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = status
                if self._store is not None:
                    self._store.complete(job_id, status=status)
                    self._store.upsert(job)

    def cancel(self, job_id: str) -> str:
        """Request cancellation.

        Returns:
            ``cancelled`` — running job marked cancelled
            ``already_cancelled`` — idempotent success
            ``already_finished`` — completed/failed (not cancelled)
            ``not_found`` — unknown job id
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job and self._store is not None:
                record = self._store.get(job_id)
                if record:
                    job = self._job_from_record(record)
                    self._jobs[job_id] = job
            if not job:
                return "not_found"
            if job.status == "cancelled":
                return "already_cancelled"
            if job.status in ("completed", "failed"):
                return "already_finished"
            job.cancel_event.set()
            job.status = "cancelled"
            if self._store is not None:
                self._store.cancel(job_id)
                self._store.upsert(job)
            return "cancelled"

    def is_cancelled(self, job_id: str | None) -> bool:
        if not job_id:
            return False
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None and self._store is not None:
                record = self._store.get(job_id)
                if record:
                    job = self._job_from_record(record)
                    self._jobs[job_id] = job
            return bool(job and job.cancel_event.is_set())


def get_job_registry(app: Any) -> TranscriptionJobRegistry:
    registry = getattr(app.state, "transcription_jobs", None)
    if registry is None:
        from .transcription_job_store import create_job_store
        from .settings import get_api_settings

        settings = get_api_settings()
        cache = getattr(app.state, "cache", None)
        redis_client = getattr(cache, "redis_client", None) if cache is not None else None
        store = create_job_store(
            state_dir=settings.state_dir,
            use_redis=settings.use_redis_cache,
            redis_client=redis_client,
        )
        registry = TranscriptionJobRegistry(store=store)
        app.state.transcription_jobs = registry
    return registry
