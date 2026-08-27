"""WebSocket router for real-time transcription logs and progress.

WebSocket `/ws/transcription` plus ticket auth.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request, WebSocket, WebSocketDisconnect

from ..dependencies import require_api_key
from ..settings import get_api_settings
from ..transcription_events import TranscriptionEventHub, get_hub
from ..ws_tickets import TICKET_TTL_SECONDS, get_ticket_store

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Transcription WebSocket"])


def _auth_ok(header_key: str | None) -> bool:
    settings = get_api_settings()
    if not settings.auth_enabled:
        return True
    return bool(header_key and header_key == settings.api_key)


def _ticket_valid(app: object, ticket: str | None) -> bool:
    """Check if a ticket exists and is not expired."""
    if not ticket:
        return False
    settings = get_api_settings()
    if not settings.auth_enabled:
        return True
    return get_ticket_store(app).valid(ticket)


@router.get("/ws/transcription/ticket")
def get_ws_ticket(
    request: Request,
    _auth: Annotated[None, Depends(require_api_key)],
) -> dict:
    """Issue a short-lived WebSocket authentication ticket.

    Browsers cannot send custom headers (X-API-Key) on WebSocket handshake.
    This endpoint (protected by normal auth) returns a token that can be
    passed as a query parameter (?token=) to the WebSocket endpoint.

    Tickets expire after 5 minutes. With ``API_USE_REDIS_CACHE=true`` the
    store is shared across uvicorn workers.
    """
    settings = get_api_settings()
    if not settings.auth_enabled:
        return {"ticket": "no-auth-required", "expires_in": TICKET_TTL_SECONDS}

    store = get_ticket_store(request.app)
    store.cleanup_expired()
    ticket = store.issue()
    return {
        "ticket": ticket,
        "expires_in": TICKET_TTL_SECONDS,
        "backend": store.backend,
    }


@router.websocket("/ws/transcription")
async def transcription_ws(
    websocket: WebSocket,
    token: Annotated[str | None, Query()] = None,
) -> None:
    """Stream transcription log/progress events (JSON).

    Authenticate with either:
        - ``X-API-Key`` header (for non-browser clients)
        - ``token`` query parameter (for browsers, ticket from /ws/transcription/ticket)

    Client may send:
        {"type": "ping"}  → server replies {"type": "pong"}
        {"type": "subscribe", "job_id": "<uuid>"}  → filter events to that job
    """
    api_key = websocket.headers.get("x-api-key")
    header_ok = _auth_ok(api_key)
    ticket_ok = _ticket_valid(websocket.app, token)

    if not (header_ok or ticket_ok):
        # Accept first so the client sees close 1008 (not a pre-handshake 403).
        await websocket.accept()
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
                sub_job = data.get("job_id")
                await hub.set_subscription(websocket, sub_job)
                await websocket.send_json(
                    {
                        "type": "subscribed",
                        "job_id": sub_job,
                    }
                )
    except WebSocketDisconnect:
        pass
    except Exception as err:
        logger.debug("WebSocket session ended: %s", err)
    finally:
        await hub.disconnect(websocket)
