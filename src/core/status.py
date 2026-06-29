"""Live process status reporting for observability and future status dashboards."""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Literal

from .logging_config import get_logger, job_id_var, log_context

StatusLevel = Literal["PHASE", "INFO", "WARN", "ERROR"]

Listener = Callable[["StatusEvent"], None]

_LEVEL_TO_LOG = {
    "PHASE": logging.INFO,
    "INFO": logging.INFO,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
}


@dataclass
class StatusEvent:
    ts: str
    level: StatusLevel
    component: str
    phase: str
    message: str
    job_id: str | None = None
    progress: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if not payload["extra"]:
            del payload["extra"]
        return payload


def _default_events_path() -> Path:
    root = os.getenv("SENTIMENT_APP_ROOT", ".")
    return Path(root) / ".cache" / "process_events.jsonl"


def _max_ring_entries() -> int:
    try:
        return max(100, int(os.getenv("SENTIMENT_STATUS_RING_SIZE", "1000")))
    except ValueError:
        return 1000


def _max_file_lines() -> int:
    try:
        return max(500, int(os.getenv("SENTIMENT_STATUS_FILE_MAX_LINES", "10000")))
    except ValueError:
        return 10000


class StatusReporter:
    """Emit live status events to logs, ring buffer, JSONL file, and listeners."""

    def __init__(
        self,
        *,
        events_path: Path | None = None,
        ring_size: int | None = None,
        file_enabled: bool | None = None,
    ) -> None:
        self._logger = get_logger("status")
        self._listeners: list[Listener] = []
        self._lock = threading.Lock()
        self._ring: deque[StatusEvent] = deque(maxlen=ring_size or _max_ring_entries())
        self._events_path = events_path or _default_events_path()
        if file_enabled is None:
            file_enabled = os.getenv("SENTIMENT_STATUS_FILE", "1").lower() not in (
                "0",
                "false",
                "no",
            )
        self._file_enabled = file_enabled

    def add_listener(self, callback: Listener) -> None:
        with self._lock:
            self._listeners.append(callback)

    def remove_listener(self, callback: Listener) -> None:
        with self._lock:
            self._listeners = [cb for cb in self._listeners if cb is not callback]

    def recent_events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._ring)
        if limit > 0:
            items = items[-limit:]
        return [event.to_dict() for event in items]

    def _emit(self, event: StatusEvent) -> None:
        with self._lock:
            self._ring.append(event)
            listeners = list(self._listeners)
        log_level = _LEVEL_TO_LOG.get(event.level, logging.INFO)
        ctx = log_context(
            job_id=event.job_id,
            component=event.component,
            phase=event.phase,
        )
        with ctx:
            msg = event.message
            if event.progress is not None:
                msg = f"{msg} ({event.progress:.0%})"
            self._logger.log(log_level, msg)
        if self._file_enabled:
            self._append_file(event)
        for listener in listeners:
            try:
                listener(event)
            except Exception:
                logging.getLogger(__name__).debug(
                    "Status listener failed", exc_info=True
                )

    def _append_file(self, event: StatusEvent) -> None:
        path = self._events_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(event.to_dict(), ensure_ascii=False) + "\n"
            with self._lock:
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(line)
                self._rotate_file_if_needed(path)
        except OSError:
            logging.getLogger(__name__).debug(
                "Could not append status event to %s", path, exc_info=True
            )

    def _rotate_file_if_needed(self, path: Path) -> None:
        max_lines = _max_file_lines()
        if not path.is_file():
            return
        try:
            with path.open(encoding="utf-8") as fh:
                line_count = sum(1 for _ in fh)
            if line_count <= max_lines:
                return
            with path.open(encoding="utf-8") as fh:
                lines = fh.readlines()
            trimmed = lines[-max_lines:]
            with path.open("w", encoding="utf-8") as fh:
                fh.writelines(trimmed)
        except OSError:
            logging.getLogger(__name__).debug(
                "Could not rotate status file %s", path, exc_info=True
            )

    def _build_event(
        self,
        level: StatusLevel,
        component: str,
        phase: str,
        message: str,
        *,
        job_id: str | None = None,
        progress: float | None = None,
        **extra: Any,
    ) -> StatusEvent:
        return StatusEvent(
            ts=datetime.now(tz=UTC).isoformat(timespec="seconds"),
            level=level,
            component=component,
            phase=phase,
            message=message,
            job_id=job_id or job_id_var.get(),
            progress=progress,
            extra=extra,
        )

    def phase(
        self,
        component: str,
        phase: str,
        message: str,
        *,
        job_id: str | None = None,
        progress: float | None = None,
        **extra: Any,
    ) -> None:
        self._emit(
            self._build_event(
                "PHASE",
                component,
                phase,
                message,
                job_id=job_id,
                progress=progress,
                **extra,
            )
        )

    def info(
        self,
        component: str,
        phase: str,
        message: str,
        *,
        job_id: str | None = None,
        progress: float | None = None,
        **extra: Any,
    ) -> None:
        self._emit(
            self._build_event(
                "INFO",
                component,
                phase,
                message,
                job_id=job_id,
                progress=progress,
                **extra,
            )
        )

    def warn(
        self,
        component: str,
        phase: str,
        message: str,
        *,
        job_id: str | None = None,
        progress: float | None = None,
        **extra: Any,
    ) -> None:
        self._emit(
            self._build_event(
                "WARN",
                component,
                phase,
                message,
                job_id=job_id,
                progress=progress,
                **extra,
            )
        )

    def error(
        self,
        component: str,
        phase: str,
        message: str,
        *,
        job_id: str | None = None,
        progress: float | None = None,
        **extra: Any,
    ) -> None:
        self._emit(
            self._build_event(
                "ERROR",
                component,
                phase,
                message,
                job_id=job_id,
                progress=progress,
                **extra,
            )
        )

    def progress(
        self,
        component: str,
        phase: str,
        processed: int,
        total: int,
        message: str | None = None,
        *,
        job_id: str | None = None,
        **extra: Any,
    ) -> None:
        total = max(total, 1)
        fraction = min(1.0, max(0.0, processed / total))
        text = message or f"{processed}/{total}"
        self._emit(
            self._build_event(
                "INFO",
                component,
                phase,
                text,
                job_id=job_id,
                progress=fraction,
                processed=processed,
                total=total,
                **extra,
            )
        )


_reporter: StatusReporter | None = None
_reporter_lock = threading.Lock()


def get_status_reporter() -> StatusReporter:
    global _reporter
    if _reporter is None:
        with _reporter_lock:
            if _reporter is None:
                _reporter = StatusReporter()
    return _reporter


def reset_status_reporter() -> None:
    """Reset global reporter (for tests)."""
    global _reporter
    with _reporter_lock:
        _reporter = None
