"""HTTP/WS tests against the real transcription WebSocket router."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from src.api.app import create_app
from src.api.settings import get_api_settings
from src.api.transcription_events import get_hub
from src.api.ws_tickets import get_ticket_store


def _ws_close_code(
    client: TestClient,
    path: str,
    headers: dict[str, str] | None = None,
) -> int:
    try:
        with client.websocket_connect(path, headers=headers or {}) as ws:
            ws.receive_json()
    except WebSocketDisconnect as exc:
        return exc.code
    raise AssertionError("WebSocket was accepted without close 1008")


@pytest.fixture(autouse=True)
def _clear_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SENTIMENT_API_KEY", raising=False)
    get_api_settings.cache_clear()


def test_ws_ticket_without_auth_and_direct_helpers() -> None:
    client = TestClient(create_app())
    issued = client.get("/ws/transcription/ticket")
    assert issued.status_code == 200
    assert issued.json()["ticket"] == "no-auth-required"
    from src.api.routers.ws_transcription import _auth_ok, _ticket_valid

    assert _auth_ok(None) is True
    assert _ticket_valid(client.app, "any") is True
    with client.websocket_connect("/ws/transcription") as ws:
        assert ws.receive_json()["type"] == "connected"
        ws.send_text("not-json")
    delattr(client.app.state, "transcription_events")
    assert get_hub(client.app) is not None


def test_ws_ticket_requires_api_key_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENTIMENT_API_KEY", "secret")
    get_api_settings.cache_clear()
    client = TestClient(create_app())
    denied = client.get("/ws/transcription/ticket")
    assert denied.status_code == 401
    assert denied.json()["error_code"] == "unauthorized"
    ok = client.get("/ws/transcription/ticket", headers={"X-API-Key": "secret"})
    assert ok.status_code == 200
    body = ok.json()
    assert body["ticket"]
    assert body["ticket"] != "no-auth-required"
    assert body["expires_in"] == 300


def test_ws_rejects_missing_and_wrong_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENTIMENT_API_KEY", "secret")
    get_api_settings.cache_clear()
    client = TestClient(create_app())
    assert _ws_close_code(client, "/ws/transcription") == 1008
    assert _ws_close_code(client, "/ws/transcription", headers={"X-API-Key": "nope"}) == 1008


def test_ws_accepts_header_key_and_ping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENTIMENT_API_KEY", "secret")
    get_api_settings.cache_clear()
    client = TestClient(create_app())
    with client.websocket_connect("/ws/transcription", headers={"X-API-Key": "secret"}) as ws:
        hello = ws.receive_json()
        assert hello["type"] == "connected"
        ws.send_json({"type": "ping"})
        assert ws.receive_json()["type"] == "pong"
        ws.send_json({"type": "subscribe", "job_id": "job-1"})
        sub = ws.receive_json()
        assert sub["type"] == "subscribed"
        assert sub["job_id"] == "job-1"


def test_ws_accepts_query_ticket(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENTIMENT_API_KEY", "secret")
    get_api_settings.cache_clear()
    client = TestClient(create_app())
    issued = client.get("/ws/transcription/ticket", headers={"X-API-Key": "secret"}).json()
    with client.websocket_connect(f"/ws/transcription?token={issued['ticket']}") as ws:
        assert ws.receive_json()["type"] == "connected"


def test_ws_rejects_expired_ticket(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENTIMENT_API_KEY", "secret")
    get_api_settings.cache_clear()
    client = TestClient(create_app())
    issued = client.get("/ws/transcription/ticket", headers={"X-API-Key": "secret"}).json()
    get_ticket_store(client.app).force_expire(issued["ticket"])
    assert _ws_close_code(client, f"/ws/transcription?token={issued['ticket']}") == 1008
