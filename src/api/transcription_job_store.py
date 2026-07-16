"""Persistent store for transcription jobs (SQLite default, optional Redis)."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

_JOB_KEY_PREFIX = "asr:jobs:"
_JOB_INDEX_KEY = "asr:jobs:index"


class JobStore(Protocol):
    """Persistence backend for :class:`~src.api.transcription_jobs.TranscriptionJob`."""

    def upsert(self, job: Any) -> None: ...

    def get(self, job_id: str) -> dict[str, Any] | None: ...

    def list(self, *, limit: int = 20) -> list[dict[str, Any]]: ...

    def complete(self, job_id: str, *, status: str = "completed") -> None: ...

    def cancel(self, job_id: str) -> None: ...


class SqliteJobStore:
    """SQLite-backed job store for single-process or shared-file deployments."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS transcription_jobs (
        job_id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        meta_json TEXT NOT NULL DEFAULT '{}',
        cancelled INTEGER NOT NULL DEFAULT 0
    )
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(self._SCHEMA)
                conn.commit()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "job_id": row["job_id"],
            "kind": row["kind"],
            "status": row["status"],
            "created_at": row["created_at"],
            "meta": json.loads(row["meta_json"] or "{}"),
            "cancelled": bool(row["cancelled"]),
        }

    def upsert(self, job: Any) -> None:
        cancelled = 1 if job.cancel_event.is_set() or job.status == "cancelled" else 0
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO transcription_jobs
                        (job_id, kind, status, created_at, meta_json, cancelled)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(job_id) DO UPDATE SET
                        kind = excluded.kind,
                        status = excluded.status,
                        created_at = excluded.created_at,
                        meta_json = excluded.meta_json,
                        cancelled = excluded.cancelled
                    """,
                    (
                        job.job_id,
                        job.kind,
                        job.status,
                        job.created_at,
                        json.dumps(dict(job.meta)),
                        cancelled,
                    ),
                )
                conn.commit()

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM transcription_jobs WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
        return self._row_to_dict(row) if row else None

    def list(self, *, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT * FROM transcription_jobs
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def complete(self, job_id: str, *, status: str = "completed") -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE transcription_jobs SET status = ? WHERE job_id = ?",
                    (status, job_id),
                )
                conn.commit()

    def cancel(self, job_id: str) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE transcription_jobs
                    SET status = 'cancelled', cancelled = 1
                    WHERE job_id = ?
                    """,
                    (job_id,),
                )
                conn.commit()

    def delete(self, job_id: str) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM transcription_jobs WHERE job_id = ?", (job_id,))
                conn.commit()


class RedisJobStore:
    """Redis-backed job store for multi-worker API deployments."""

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client

    def _job_key(self, job_id: str) -> str:
        return f"{_JOB_KEY_PREFIX}{job_id}"

    @staticmethod
    def _encode(job: Any) -> str:
        return json.dumps(
            {
                "job_id": job.job_id,
                "kind": job.kind,
                "status": job.status,
                "created_at": job.created_at,
                "meta": dict(job.meta),
                "cancelled": job.cancel_event.is_set() or job.status == "cancelled",
            }
        )

    @staticmethod
    def _decode(raw: str) -> dict[str, Any]:
        data = json.loads(raw)
        data["meta"] = dict(data.get("meta") or {})
        data["cancelled"] = bool(data.get("cancelled"))
        return data

    def upsert(self, job: Any) -> None:
        key = self._job_key(job.job_id)
        self._redis.set(key, self._encode(job))
        try:
            score = datetime.fromisoformat(job.created_at).timestamp()
        except ValueError:
            score = datetime.now().timestamp()
        self._redis.zadd(_JOB_INDEX_KEY, {job.job_id: score})

    def get(self, job_id: str) -> dict[str, Any] | None:
        raw = self._redis.get(self._job_key(job_id))
        if not raw:
            return None
        return self._decode(raw)

    def list(self, *, limit: int = 20) -> list[dict[str, Any]]:
        job_ids = self._redis.zrevrange(_JOB_INDEX_KEY, 0, max(limit - 1, 0))
        jobs: list[dict[str, Any]] = []
        for job_id in job_ids:
            record = self.get(job_id)
            if record:
                jobs.append(record)
        return jobs

    def complete(self, job_id: str, *, status: str = "completed") -> None:
        record = self.get(job_id)
        if not record:
            return
        record["status"] = status
        self._redis.set(self._job_key(job_id), json.dumps(record))

    def cancel(self, job_id: str) -> None:
        record = self.get(job_id)
        if not record:
            return
        record["status"] = "cancelled"
        record["cancelled"] = True
        self._redis.set(self._job_key(job_id), json.dumps(record))

    def delete(self, job_id: str) -> None:
        self._redis.delete(self._job_key(job_id))
        self._redis.zrem(_JOB_INDEX_KEY, job_id)


def create_job_store(
    *,
    state_dir: str | Path,
    use_redis: bool = False,
    redis_client: Any | None = None,
) -> JobStore:
    """Build the active job store (Redis when configured, else SQLite under state_dir)."""
    if use_redis and redis_client is not None:
        try:
            redis_client.ping()
            return RedisJobStore(redis_client)
        except Exception as exc:
            logger.warning("Redis job store unavailable, using SQLite: %s", exc)
    db_path = Path(state_dir) / "transcription_jobs.db"
    return SqliteJobStore(db_path)
