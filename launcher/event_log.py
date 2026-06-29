"""Thread-safe activity log for the launcher GUI."""

from __future__ import annotations

import queue
import re
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

EventLevel = Literal["PHASE", "INFO", "WARN", "ERROR"]

_LINE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}:\d{2}\.\d{3})"
    r" (PHASE|INFO |WARN |ERROR) (?:\[([^\]]+)\] )?(.*)$"
)


@dataclass(frozen=True)
class LogEvent:
    timestamp: str
    level: EventLevel
    phase: str
    message: str

    def format_line(self) -> str:
        phase = f"[{self.phase}] " if self.phase else ""
        return f"{self.timestamp} {self.level:5} {phase}{self.message}"


def _timestamp_now(*, full: bool) -> str:
    now = datetime.now()
    ms = now.microsecond // 1000
    if full:
        return now.strftime("%Y-%m-%d %H:%M:%S.") + f"{ms:03d}"
    return now.strftime("%H:%M:%S.") + f"{ms:03d}"


def _parse_log_line(line: str) -> LogEvent | None:
    match = _LINE_RE.match(line.strip())
    if not match:
        return None
    ts, level_raw, phase, message = match.groups()
    level = level_raw.strip()  # type: ignore[assignment]
    if level not in ("PHASE", "INFO", "WARN", "ERROR"):
        return None
    return LogEvent(timestamp=ts, level=level, phase=phase or "", message=message)


class EventLog:
    """Ring buffer of events with optional file persistence and GUI queue."""

    def __init__(
        self,
        max_entries: int = 200,
        *,
        log_path: Path | None = None,
        load_max_lines: int = 500,
    ) -> None:
        self._max = max(1, max_entries)
        self._entries: deque[LogEvent] = deque(maxlen=self._max)
        self._queue: queue.Queue[LogEvent] = queue.Queue()
        self._log_path = log_path
        self._file_lock = threading.Lock()
        if log_path is not None:
            self.load_recent(log_path, max_lines=load_max_lines)

    @property
    def log_path(self) -> Path | None:
        return self._log_path

    def append(
        self,
        level: EventLevel,
        message: str,
        *,
        phase: str = "",
    ) -> LogEvent:
        ts_file = _timestamp_now(full=True)
        ts_gui = ts_file[11:] if len(ts_file) > 11 else ts_file
        event = LogEvent(timestamp=ts_gui, level=level, phase=phase, message=message)
        self._entries.append(event)
        self._queue.put(event)
        if self._log_path is not None:
            line = LogEvent(timestamp=ts_file, level=level, phase=phase, message=message)
            self._append_to_file(line.format_line())
        self._emit_to_status_reporter(level, phase, message)
        return event

    def _emit_to_status_reporter(self, level: EventLevel, phase: str, message: str) -> None:
        try:
            from src.core.status import get_status_reporter

            reporter = get_status_reporter()
            component = "launcher"
            if level == "PHASE":
                reporter.phase(component, phase or "general", message)
            elif level == "WARN":
                reporter.warn(component, phase or "general", message)
            elif level == "ERROR":
                reporter.error(component, phase or "general", message)
            else:
                reporter.info(component, phase or "general", message)
        except Exception:
            pass

    def _append_to_file(self, line: str) -> None:
        if self._log_path is None:
            return
        with self._file_lock:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def load_recent(self, path: Path, *, max_lines: int = 500) -> int:
        """Load tail of log file into ring buffer (no GUI queue). Returns count loaded."""
        if not path.is_file():
            return 0
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        tail = lines[-max_lines:] if max_lines > 0 else lines
        loaded: list[LogEvent] = []
        for line in tail:
            event = _parse_log_line(line)
            if event is not None:
                gui_ts = event.timestamp[11:] if event.timestamp[:4].isdigit() else event.timestamp
                loaded.append(
                    LogEvent(
                        timestamp=gui_ts,
                        level=event.level,
                        phase=event.phase,
                        message=event.message,
                    )
                )
        self._entries.clear()
        for event in loaded[-self._max :]:
            self._entries.append(event)
        return len(loaded)

    def phase(self, phase: str, message: str) -> LogEvent:
        return self.append("PHASE", message, phase=phase)

    def info(self, message: str, *, phase: str = "") -> LogEvent:
        return self.append("INFO", message, phase=phase)

    def warn(self, message: str, *, phase: str = "") -> LogEvent:
        return self.append("WARN", message, phase=phase)

    def error(self, message: str, *, phase: str = "") -> LogEvent:
        return self.append("ERROR", message, phase=phase)

    def entries(self) -> list[LogEvent]:
        return list(self._entries)

    def clear(self) -> None:
        """Clear in-memory buffer and pending GUI queue; log file is untouched."""
        self._entries.clear()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def poll_queue(self) -> list[LogEvent]:
        batch: list[LogEvent] = []
        while True:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return batch
