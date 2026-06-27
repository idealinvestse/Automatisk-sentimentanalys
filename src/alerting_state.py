"""Persistent alerting state for multi-request / multi-worker visibility.

Stores circuit breaker counters in a JSON file so dashboard status survives
process restarts within a single host. For true multi-worker consistency use Redis.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(".cache/alerting_state.json")


class AlertingStateManager:
    """Thread-safe JSON-backed store for webhook circuit breaker state."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else _DEFAULT_PATH
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"consecutive_failures": 0, "circuit_breaker_open": False}
        try:
            with self.path.open(encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            logger.warning("Failed to read alerting state %s: %s", self.path, exc)
            return {"consecutive_failures": 0, "circuit_breaker_open": False}

    def _write(self, data: dict[str, Any]) -> None:
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        tmp.replace(self.path)

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._read())

    def record_failure(self, threshold: int = 5) -> dict[str, Any]:
        with self._lock:
            data = self._read()
            failures = int(data.get("consecutive_failures", 0)) + 1
            data["consecutive_failures"] = failures
            if failures >= threshold:
                data["circuit_breaker_open"] = True
            self._write(data)
            return dict(data)

    def record_success(self) -> dict[str, Any]:
        with self._lock:
            data = self._read()
            data["consecutive_failures"] = 0
            data["circuit_breaker_open"] = False
            self._write(data)
            return dict(data)

    def reset(self) -> dict[str, Any]:
        with self._lock:
            data = {"consecutive_failures": 0, "circuit_breaker_open": False}
            self._write(data)
            return dict(data)

    def sync_to_engine(self, engine: Any) -> None:
        """Apply persisted circuit breaker flags onto an AlertEngine instance."""
        data = self.get_status()
        if data.get("circuit_breaker_open"):
            engine._webhook_disabled = True
            engine._consecutive_failures = int(data.get("consecutive_failures", 0))

    def sync_from_engine(self, engine: Any) -> None:
        """Persist engine circuit breaker counters to disk."""
        with self._lock:
            data = self._read()
            data["consecutive_failures"] = getattr(engine, "_consecutive_failures", 0)
            data["circuit_breaker_open"] = getattr(engine, "_webhook_disabled", False)
            self._write(data)