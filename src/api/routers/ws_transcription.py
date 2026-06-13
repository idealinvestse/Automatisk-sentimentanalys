"""WebSocket router for real-time transcription logs and progress.

Fas 3 WebSocket – docs/MIGRATION_TO_NICEGUI_PLAN.md §3
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from ..settings import get_api_settings
from ..transcription_events import TranscriptionEventHub, get_hub

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Transcription WebSocket"])


def _auth_ok(api_key: str | None) -> bool:
    settings = get_api_settings()
    if not settings.auth_enabled:
        return True
    return bool(api_key and api_key == settings.api_key)


@router.websocket("/ws/transcription")
async def transcription_ws(
    websocket: WebSocket,
    api_key: str | None = Query(default=None),
) -> None:
    """Stream transcription log/progress events (JSON).

    Client may send:
        {"type": "ping"}  → server replies {"type": "pong"}
        {"type": "subscribe", "job_id": "<uuid>"}  → optional filter hint (server still broadcasts all)
    """
    if not _auth_ok(api_key):
        await websocket.close(code=1008, reason="Unauthorized")
        return

    hub: TranscriptionEventHub = get_hub(websocket.app)
    await hub.connect(websocket)
    await websocket.send_json({"type": "connected", "msg": "Transcription WebSocket ready"})

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")
            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
            elif msg_type == "subscribe":
                await websocket.send_json(
                    {
                        "type": "subscribed",
                        "job_id": data.get("job_id"),
                    }
                )
    except WebSocketDisconnect:
        pass
    except Exception as err:
        logger.debug("WebSocket session ended: %s", err)
    finally:
        await hub.disconnect(websocket)