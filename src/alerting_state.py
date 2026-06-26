"""Alerting state backend for circuit breaker.

Supports in-memory (default) and optional Redis for multi-worker consistency.

This is a pragmatic implementation for TASK-07.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class AlertingState:
    """Abstract-ish state holder for circuit breaker.

    In-memory by default. Redis backend can be passed in.
    """

    def __init__(self, redis_client: Any | None = None, key_prefix: str = "alerting:"):
        self.redis = redis_client
        self.key_prefix = key_prefix
        self._in_memory: dict[str, Any] = {
            "consecutive_failures": 0,
            "circuit_breaker_open": False,
        }

    def _key(self, name: str) -> str:
        return f"{self.key_prefix}{name}"

    def get_consecutive_failures(self) -> int:
        if self.redis:
            try:
                val = self.redis.get(self._key("consecutive_failures"))
                return int(val) if val else 0
            except Exception as e:
                logger.warning("Redis read failed for failures: %s", e)
                return self._in_memory.get("consecutive_failures", 0)
        return self._in_memory.get("consecutive_failures", 0)

    def set_consecutive_failures(self, value: int) -> None:
        if self.redis:
            try:
                self.redis.set(self._key("consecutive_failures"), value)
                return
            except Exception as e:
                logger.warning("Redis write failed for failures: %s", e)
        self._in_memory["consecutive_failures"] = value

    def is_circuit_breaker_open(self) -> bool:
        if self.redis:
            try:
                val = self.redis.get(self._key("circuit_breaker_open"))
                return str(val).lower() in ("1", "true", "yes") if val else False
            except Exception as e:
                logger.warning("Redis read failed for breaker: %s", e)
                return self._in_memory.get("circuit_breaker_open", False)
        return self._in_memory.get("circuit_breaker_open", False)

    def set_circuit_breaker_open(self, value: bool) -> None:
        if self.redis:
            try:
                self.redis.set(self._key("circuit_breaker_open"), "1" if value else "0")
                return
            except Exception as e:
                logger.warning("Redis write failed for breaker: %s", e)
        self._in_memory["circuit_breaker_open"] = value

    def increment_failures(self) -> int:
        current = self.get_consecutive_failures() + 1
        self.set_consecutive_failures(current)
        return current

    def reset(self) -> None:
        self.set_consecutive_failures(0)
        self.set_circuit_breaker_open(False)