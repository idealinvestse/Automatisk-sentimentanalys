"""Thread-safe activity log for the launcher GUI."""

from __future__ import annotations

import queue
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

EventLevel = Literal["PHASE", "INFO", "WARN", "ERROR"]


@dataclass(frozen=True)
class LogEvent:
    timestamp: str
    level: EventLevel
    phase: str
    message: str

    def format_line(self) -> str:
        phase = f"[{self.phase}] " if self.phase else ""
        return f"{self.timestamp} {self.level:5} {phase}{self.message}"


class EventLog:
    """Ring buffer of events with a queue for cross-thread GUI delivery."""

    def __init__(self, max_entries: int = 200) -> None:
        self._max = max(1, max_entries)
        self._entries: deque[LogEvent] = deque(maxlen=self._max)
        self._queue: queue.Queue[LogEvent] = queue.Queue()

    def append(
        self,
        level: EventLevel,
        message: str,
        *,
        phase: str = "",
    ) -> LogEvent:
        ts = datetime.now().strftime("%H:%M:%S.") + f"{datetime.now().microsecond // 1000:03d}"
        event = LogEvent(timestamp=ts, level=level, phase=phase, message=message)
        self._entries.append(event)
        self._queue.put(event)
        return event

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