"""Unit tests for WebSocket ticket store (memory + Redis-shaped client)."""

from __future__ import annotations

from src.api.ws_tickets import TicketStore


class _FakeRedis:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.data[key] = value

    def exists(self, key: str) -> int:
        return 1 if key in self.data else 0

    def delete(self, key: str) -> None:
        self.data.pop(key, None)


def test_memory_ticket_roundtrip() -> None:
    store = TicketStore()
    ticket = store.issue()
    assert store.backend == "memory"
    assert store.valid(ticket) is True
    assert store.valid("missing") is False
    store.force_expire(ticket)
    assert store.valid(ticket) is False


def test_redis_ticket_roundtrip() -> None:
    fake = _FakeRedis()
    store = TicketStore(redis_client=fake)
    ticket = store.issue()
    assert store.backend == "redis"
    assert store.valid(ticket) is True
    assert any(k.endswith(ticket) for k in fake.data)
    store.force_expire(ticket)
    assert store.valid(ticket) is False
