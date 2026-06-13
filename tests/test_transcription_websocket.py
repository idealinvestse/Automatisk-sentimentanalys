"""Tests for transcription WebSocket hub and dashboard client (Fas 3)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.nicegui_dashboard.services.nicegui_api_client import JOB_HEADER, NiceGUIAPIClient
from app.nicegui_dashboard.services.transcription_service import TranscriptionState
from app.nicegui_dashboard.services.transcription_ws_client import TranscriptionWSListener
from src.api import app
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
    assert listener._ws_url() == "ws://localhost:8000/ws/transcription?api_key=k"


def test_get_hub_on_app() -> None:
    hub = get_hub(app)
    assert isinstance(hub, TranscriptionEventHub)