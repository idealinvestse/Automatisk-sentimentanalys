"""Live process status reporting for observability and future status dashboards."""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import threading
import traceback
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from .logging_config import get_logger, job_id_var, log_context
from .metrics import record_status_event

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
    error_code: str | None = None
    exception_type: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload.get("error_code") is None:
            del payload["error_code"]
        if payload.get("exception_type") is None:
            del payload["exception_type"]
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


def _dedup_window_s() -> float:
    try:
        return max(0.0, float(os.getenv("SENTIMENT_STATUS_DEDUP_WINDOW_S", "0")))
    except ValueError:
        return 0.0


def _file_max_bytes() -> int:
    try:
        return max(1024, int(os.getenv("SENTIMENT_STATUS_FILE_MAX_BYTES", str(5 * 1024 * 1024))))
    except ValueError:
        return 5 * 1024 * 1024


def _setup_file_logger(path: Path) -> logging.Logger:
    """Dedicated logger with RotatingFileHandler (no hot-path ring lock)."""
    logger_name = f"status.file.{path.resolve()}"
    file_logger = logging.getLogger(logger_name)
    if file_logger.handlers:
        return file_logger
    file_logger.setLevel(logging.INFO)
    file_logger.propagate = False
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        path,
        maxBytes=_file_max_bytes(),
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    file_logger.addHandler(handler)
    return file_logger


def derive_job_status(events: list[dict[str, Any]], job_id: str) -> dict[str, Any]:
    """Derive live job summary from status events."""
    job_events = [e for e in events if e.get("job_id") == job_id]
    if not job_events:
        return {"job_id": job_id, "found": False}

    current_phase = job_events[-1].get("phase")
    current_component = job_events[-1].get("component")
    progress = next(
        (e.get("progress") for e in reversed(job_events) if e.get("progress") is not None), None
    )
    last_error = next(
        (e for e in reversed(job_events) if e.get("level") == "ERROR"),
        None,
    )
    started = job_events[0].get("ts")
    return {
        "job_id": job_id,
        "found": True,
        "current_phase": current_phase,
        "current_component": current_component,
        "progress": progress,
        "started": started,
        "last_event": job_events[-1],
        "last_error": last_error,
        "event_count": len(job_events),
    }


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
        self._file_logger: logging.Logger | None = None
        if self._file_enabled:
            self._file_logger = _setup_file_logger(self._events_path)
        self._dedup_window = _dedup_window_s()
        self._dedup_key: tuple[str, str, str] | None = None
        self._dedup_count = 0
        self._dedup_first_ts: str | None = None

    def add_listener(self, callback: Listener) -> None:
        with self._lock:
            self._listeners.append(callback)

    def remove_listener(self, callback: Listener) -> None:
        with self._lock:
            self._listeners = [cb for cb in self._listeners if cb is not callback]

    def recent_events(
        self,
        limit: int = 100,
        *,
        job_id: str | None = None,
        component: str | None = None,
        level: str | None = None,
        since: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._ring)
        if job_id:
            items = [e for e in items if e.job_id == job_id]
        if component:
            items = [e for e in items if e.component == component]
        if level:
            items = [e for e in items if e.level == level.upper()]
        if since:
            items = [e for e in items if e.ts >= since]
        if limit > 0:
            items = items[-limit:]
        return [event.to_dict() for event in items]

    def _flush_dedup(self) -> None:
        if self._dedup_count > 1 and self._dedup_key is not None:
            comp, ph, msg = self._dedup_key
            summary = self._build_event(
                "INFO",
                comp,
                ph,
                f"{msg} (upprepad {self._dedup_count} gånger)",
                dedup_first_ts=self._dedup_first_ts,
            )
            self._emit_direct(summary)
        self._dedup_count = 0
        self._dedup_key = None
        self._dedup_first_ts = None

    def _emit(self, event: StatusEvent) -> None:
        if self._dedup_window <= 0:
            self._emit_direct(event)
            return
        key = (event.component, event.phase, event.message)
        if self._dedup_key == key and self._dedup_count >= 1:
            self._dedup_count += 1
            return
        self._flush_dedup()
        self._dedup_key = key
        self._dedup_count = 1
        self._dedup_first_ts = event.ts
        self._emit_direct(event)

    def _emit_direct(self, event: StatusEvent) -> None:
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
        record_status_event(event.level, event.component, event.error_code or "")
        if self._file_enabled and self._file_logger is not None:
            self._file_logger.info(json.dumps(event.to_dict(), ensure_ascii=False))
        for listener in listeners:
            try:
                listener(event)
            except Exception:
                logging.getLogger(__name__).debug("Status listener failed", exc_info=True)

    def _build_event(
        self,
        level: StatusLevel,
        component: str,
        phase: str,
        message: str,
        *,
        job_id: str | None = None,
        progress: float | None = None,
        error_code: str | None = None,
        exc: BaseException | None = None,
        **extra: Any,
    ) -> StatusEvent:
        exception_type: str | None = None
        if exc is not None:
            exception_type = type(exc).__name__
            if error_code is None and hasattr(exc, "error_code"):
                error_code = str(exc.error_code)
            tb = traceback.format_exc()
            if tb and tb.strip() != "NoneType: None":
                extra.setdefault("traceback_tail", tb[-500:])
        return StatusEvent(
            ts=datetime.now(tz=UTC).isoformat(timespec="seconds"),
            level=level,
            component=component,
            phase=phase,
            message=message,
            job_id=job_id or job_id_var.get(),
            progress=progress,
            error_code=error_code,
            exception_type=exception_type,
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
        error_code: str | None = None,
        exc: BaseException | None = None,
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
                error_code=error_code,
                exc=exc,
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
        error_code: str | None = None,
        exc: BaseException | None = None,
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
                error_code=error_code,
                exc=exc,
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
        error_code: str | None = None,
        exc: BaseException | None = None,
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
                error_code=error_code,
                exc=exc,
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
        error_code: str | None = None,
        exc: BaseException | None = None,
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
                error_code=error_code or "phase_failed",
                exc=exc,
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
        if _reporter is not None and _reporter._dedup_window > 0:
            _reporter._flush_dedup()
        _reporter = None
