"""WebSocket router for real-time transcription logs and progress.

Fas 3 WebSocket – docs/archive/MIGRATION_TO_NICEGUI_PLAN.md §3
"""

from __future__ import annotations

import logging
import secrets
import time
from typing import Annotated

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect

from ..dependencies import require_api_key
from ..router_errors import run_route
from ..settings import get_api_settings
from ..transcription_events import TranscriptionEventHub, get_hub

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Transcription WebSocket"])

# Simple in-memory ticket store with expiration (5 minutes)
# In production with multiple workers, use Redis or similar
_tickets: dict[str, float] = {}
_TICKET_TTL = 300  # 5 minutes


def _auth_ok(header_key: str | None) -> bool:
    settings = get_api_settings()
    if not settings.auth_enabled:
        return True
    return bool(header_key and header_key == settings.api_key)


def _ticket_valid(ticket: str | None) -> bool:
    """Check if a ticket exists and is not expired."""
    if not ticket:
        return False
    settings = get_api_settings()
    if not settings.auth_enabled:
        return True
    now = time.time()
    expiry = _tickets.get(ticket)
    if expiry is None:
        return False
    if now > expiry:
        # Clean up expired ticket
        _tickets.pop(ticket, None)
        return False
    return True


def _cleanup_expired_tickets() -> None:
    """Remove expired tickets from the store."""
    now = time.time()
    expired = [t for t, exp in _tickets.items() if exp < now]
    for t in expired:
        _tickets.pop(t, None)


@router.get("/ws/transcription/ticket")
def get_ws_ticket(_auth: Annotated[None, Depends(require_api_key)]) -> dict:
    """Issue a short-lived WebSocket authentication ticket.

    Browsers cannot send custom headers (X-API-Key) on WebSocket handshake.
    This endpoint (protected by normal auth) returns a one-time token that
    can be passed as a query parameter (?token=) to the WebSocket endpoint.

    Tickets expire after 5 minutes.
    """
    settings = get_api_settings()
    if not settings.auth_enabled:
        # No auth needed, return a dummy ticket
        return {"ticket": "no-auth-required", "expires_in": _TICKET_TTL}

    _cleanup_expired_tickets()
    ticket = secrets.token_urlsafe(32)
    _tickets[ticket] = time.time() + _TICKET_TTL
    return {"ticket": ticket, "expires_in": _TICKET_TTL}


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
    # Try header auth first, then ticket auth
    api_key = websocket.headers.get("x-api-key")
    header_ok = _auth_ok(api_key)
    ticket_ok = _ticket_valid(token)

    if not (header_ok or ticket_ok):
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
