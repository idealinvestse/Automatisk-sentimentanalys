"""WebSocket auth ticket store — memory by default, Redis when available.

Multi-worker deployments must set ``API_USE_REDIS_CACHE=true`` (and ``REDIS_URL``)
so tickets issued by one uvicorn worker are accepted by another.
"""

from __future__ import annotations

import logging
import secrets
import time
from typing import Any

logger = logging.getLogger(__name__)

TICKET_TTL_SECONDS = 300
_TICKET_KEY_PREFIX = "ws:ticket:"


class TicketStore:
    """Issue and validate short-lived WS tickets."""

    def __init__(
        self,
        *,
        redis_client: Any | None = None,
        ttl_seconds: int = TICKET_TTL_SECONDS,
    ) -> None:
        self._redis = redis_client
        self._ttl = ttl_seconds
        self._memory: dict[str, float] = {}

    @property
    def backend(self) -> str:
        return "redis" if self._redis is not None else "memory"

    def issue(self) -> str:
        ticket = secrets.token_urlsafe(32)
        if self._redis is not None:
            try:
                self._redis.setex(f"{_TICKET_KEY_PREFIX}{ticket}", self._ttl, "1")
                return ticket
            except Exception as exc:
                logger.warning("Redis ticket issue failed, using memory: %s", exc)
        self._memory[ticket] = time.time() + self._ttl
        return ticket

    def valid(self, ticket: str | None) -> bool:
        if not ticket:
            return False
        if self._redis is not None:
            try:
                return bool(self._redis.exists(f"{_TICKET_KEY_PREFIX}{ticket}"))
            except Exception as exc:
                logger.warning("Redis ticket check failed, falling back to memory: %s", exc)
        expiry = self._memory.get(ticket)
        if expiry is None:
            return False
        if time.time() > expiry:
            self._memory.pop(ticket, None)
            return False
        return True

    def force_expire(self, ticket: str) -> None:
        """Test helper: mark a ticket expired in the active backend."""
        if self._redis is not None:
            try:
                self._redis.delete(f"{_TICKET_KEY_PREFIX}{ticket}")
            except Exception:
                pass
        self._memory[ticket] = 0.0

    def cleanup_expired(self) -> None:
        """Drop expired in-memory tickets (Redis TTLs are automatic)."""
        now = time.time()
        expired = [t for t, exp in self._memory.items() if exp < now]
        for t in expired:
            self._memory.pop(t, None)


def get_ticket_store(app: Any) -> TicketStore:
    store = getattr(app.state, "ws_tickets", None)
    if store is None:
        store = TicketStore()
        app.state.ws_tickets = store
    return store
