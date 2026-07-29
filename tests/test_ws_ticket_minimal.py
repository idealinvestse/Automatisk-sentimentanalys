"""Minimal WebSocket smoke without full pipeline import (Fas 6 hardening).

Uses a stub FastAPI app that only exercises connect/ping. Auth rejection is
covered by webui e2e / API WS tests against the real router.
"""

from __future__ import annotations

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocket, WebSocketDisconnect


def get_ws_router() -> APIRouter:
    """Create a minimal WebSocket router for connect/ping smoke tests."""
    router = APIRouter(tags=["Transcription"])

    @router.get("/ws/transcription/ticket")
    async def ws_ticket() -> dict:
        return {"ticket": "no-auth-required", "expires_in": 300}

    @router.websocket("/ws/transcription")
    async def ws_transcription(websocket: WebSocket, token: str | None = None) -> None:
        await websocket.accept()
        await websocket.send_json({"type": "connected"})
        try:
            while True:
                data = await websocket.receive_json()
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
        except WebSocketDisconnect:
            pass

    return router


def create_minimal_app() -> FastAPI:
    app = FastAPI()
    app.include_router(get_ws_router(), prefix="/api")
    return app


def test_ws_ticket_endpoint_returns_ticket() -> None:
    client = TestClient(create_minimal_app())
    response = client.get("/api/ws/transcription/ticket")
    assert response.status_code == 200
    data = response.json()
    assert data["ticket"] == "no-auth-required"
    assert data["expires_in"] == 300


def test_ws_accepts_no_auth() -> None:
    client = TestClient(create_minimal_app())
    with client.websocket_connect("/api/ws/transcription") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "connected"
        ws.send_json({"type": "ping"})
        pong = ws.receive_json()
        assert pong["type"] == "pong"


def test_ws_connect_with_token_query_still_works_in_stub() -> None:
    """Stub accepts any token; auth rejection belongs to full API WS tests."""
    client = TestClient(create_minimal_app())
    with client.websocket_connect("/api/ws/transcription?token=any") as ws:
        assert ws.receive_json()["type"] == "connected"
