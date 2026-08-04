"""Cross-worker Redis pub/sub for TranscriptionEventHub (lightweight mock)."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from src.api.transcription_events import TranscriptionEventHub, _REDIS_CHANNEL


class _FakePubSub:
    def __init__(self, bus: "_FakeRedis") -> None:
        self._bus = bus
        self._queue: list[dict[str, Any]] = []
        self._subscribed = False

    def subscribe(self, channel: str) -> None:
        self._subscribed = True
        self._bus._subscribers.setdefault(channel, []).append(self)

    def get_message(self, timeout: float = 1.0) -> dict[str, Any] | None:  # noqa: ARG002
        if self._queue:
            return self._queue.pop(0)
        return None

    def unsubscribe(self, channel: str) -> None:  # noqa: ARG002
        self._subscribed = False

    def close(self) -> None:
        return None


class _FakeRedis:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[_FakePubSub]] = {}
        self._lock = threading.Lock()

    def publish(self, channel: str, data: str) -> int:
        with self._lock:
            subs = list(self._subscribers.get(channel, []))
        for sub in subs:
            sub._queue.append({"type": "message", "data": data, "channel": channel})
        return len(subs)

    def pubsub(self, ignore_subscribe_messages: bool = True) -> _FakePubSub:  # noqa: ARG002
        return _FakePubSub(self)

    def ping(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_hub_publishes_and_receives_via_redis() -> None:
    redis = _FakeRedis()
    received: list[dict] = []

    class _FakeWs:
        async def accept(self) -> None:
            return None

        async def send_json(self, data):  # noqa: ANN001
            received.append(data)

    hub_a = TranscriptionEventHub(redis_client=redis)
    hub_b = TranscriptionEventHub(redis_client=redis)
    loop = asyncio.get_running_loop()
    hub_a.bind_loop(loop)
    hub_b.bind_loop(loop)
    await hub_b.start_redis_listener()
    await hub_b.connect(_FakeWs())  # type: ignore[arg-type]

    assert hub_a.backend == "redis"
    assert hub_b.backend == "redis"
    assert _REDIS_CHANNEL

    # Wait until listener has subscribed (emit before subscribe drops the message)
    for _ in range(50):
        if redis._subscribers.get(_REDIS_CHANNEL):
            break
        await asyncio.sleep(0.05)
    assert redis._subscribers.get(_REDIS_CHANNEL), "listener never subscribed"

    hub_a.log(job_id="j1", level="INFO", msg="hello-from-a")

    for _ in range(100):
        if received:
            break
        await asyncio.sleep(0.05)

    await hub_b.stop_redis_listener()
    assert any(e.get("msg") == "hello-from-a" for e in received)
