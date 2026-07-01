"""Tests for transcription WebSocket hub and dashboard client (Fas 3)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.archive.nicegui_dashboard.services.nicegui_api_client import JOB_HEADER, NiceGUIAPIClient
from app.archive.nicegui_dashboard.services.transcription_service import TranscriptionState
from app.archive.nicegui_dashboard.services.transcription_ws_client import TranscriptionWSListener
from src.api import app
from src.api.app import create_app
from src.api.settings import get_api_settings
from src.api.transcription_events import TranscriptionEventHub, get_hub


@pytest.fixture(autouse=True)
def _clear_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SENTIMENT_API_KEY", raising=False)
    get_api_settings.cache_clear()


def test_event_hub_log_and_progress() -> None:
    hub = TranscriptionEventHub()
    received: list[dict] = []

    class FakeWS:
        async def accept(self) -> None:
            return None

        async def send_json(self, data: dict) -> None:
            received.append(data)

    async def _run() -> None:
        import asyncio

        hub.bind_loop(asyncio.get_running_loop())
        ws = FakeWS()
        await hub.connect(ws)
        hub.log(job_id="job-1", level="INFO", msg="Hej", file="a.wav")
        hub.progress(job_id="job-1", processed=1, total=3, current_file="a.wav", progress=0.33)
        await asyncio.sleep(0.05)

    import asyncio

    asyncio.run(_run())
    assert any(e.get("type") == "log" and e.get("msg") == "Hej" for e in received)
    assert any(e.get("type") == "progress" and e.get("processed") == 1 for e in received)


def test_ws_transcription_endpoint_connects() -> None:
    client = TestClient(app)
    with client.websocket_connect("/ws/transcription") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "connected"
        ws.send_json({"type": "ping"})
        pong = ws.receive_json()
        assert pong["type"] == "pong"


def test_api_client_job_header() -> None:
    client = NiceGUIAPIClient("http://localhost:8000", api_key="secret")
    client.set_job_id("abc-123")
    headers = client._headers()
    assert headers[JOB_HEADER] == "abc-123"
    assert headers["X-API-Key"] == "secret"


def test_transcription_state_apply_ws_event() -> None:
    state = TranscriptionState()
    events: list[str] = []
    state.add_listener(lambda t, _p: events.append(t))

    state.apply_ws_event(
        {"type": "log", "job_id": "j1", "level": "INFO", "msg": "Server log", "file": "x.wav"}
    )
    state.apply_ws_event(
        {"type": "progress", "job_id": "j1", "processed": 2, "total": 5, "progress": 0.4}
    )
    state.apply_ws_event({"type": "connected"})

    assert state.logs[-1]["msg"] == "Server log"
    assert state.logs[-1]["source"] == "ws"
    assert state.status["processed"] == 2
    assert state.ws_status == "connected"
    assert "log" in events
    assert "progress" in events
    assert "ws" in events


def test_ws_listener_url_builds() -> None:
    client = NiceGUIAPIClient("http://localhost:8000", api_key="k")
    listener = TranscriptionWSListener(client, on_event=lambda _e: None)
    assert listener._ws_url() == "ws://localhost:8000/ws/transcription"
    assert listener._ws_headers() == {"X-API-Key": "k"}


def test_ws_transcription_rejects_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENTIMENT_API_KEY", "secret-key")
    get_api_settings.cache_clear()
    authed_client = TestClient(create_app())
    with pytest.raises(WebSocketDisconnect), authed_client.websocket_connect("/ws/transcription"):
        pass


def test_ws_transcription_accepts_header_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENTIMENT_API_KEY", "secret-key")
    get_api_settings.cache_clear()
    authed_client = TestClient(create_app())
    with authed_client.websocket_connect(
        "/ws/transcription",
        headers={"X-API-Key": "secret-key"},
    ) as ws:
        msg = ws.receive_json()
        assert msg["type"] == "connected"


def test_get_hub_on_app() -> None:
    hub = get_hub(app)
    assert isinstance(hub, TranscriptionEventHub)


def test_ws_subscription_filters_events() -> None:
    hub = TranscriptionEventHub()
    received_a: list[dict] = []
    received_b: list[dict] = []

    class FakeWS:
        def __init__(self, bucket: list[dict]) -> None:
            self.bucket = bucket

        async def accept(self) -> None:
            return None

        async def send_json(self, data: dict) -> None:
            self.bucket.append(data)

    async def _run() -> None:
        import asyncio

        hub.bind_loop(asyncio.get_running_loop())
        ws_a = FakeWS(received_a)
        ws_b = FakeWS(received_b)
        await hub.connect(ws_a)
        await hub.connect(ws_b)
        await hub.set_subscription(ws_a, "job-a")
        await hub.set_subscription(ws_b, "job-b")
        hub.log(job_id="job-a", level="INFO", msg="for A")
        hub.log(job_id="job-b", level="INFO", msg="for B")
        await asyncio.sleep(0.05)

    import asyncio

    asyncio.run(_run())
    assert any(e.get("msg") == "for A" for e in received_a)
    assert not any(e.get("msg") == "for B" for e in received_a)
    assert any(e.get("msg") == "for B" for e in received_b)
    assert not any(e.get("msg") == "for A" for e in received_b)


def test_asr_payload_includes_beam_size() -> None:
    from app.archive.nicegui_dashboard.services.nicegui_api_client import _asr_payload

    payload = _asr_payload({"beam_size": 7, "preprocess": True, "hotwords": "a, b"})
    assert payload["beam_size"] == 7
    assert payload["preprocess"] is True
    assert payload["hotwords"] == ["a", "b"]
