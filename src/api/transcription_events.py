"""Real-time transcription event hub for WebSocket clients.

Fas 3 WebSocket – docs/MIGRATION_TO_NICEGUI_PLAN.md §3
Thread-safe emit from batch workers; async broadcast to subscribers.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)

JOB_HEADER = "X-Transcription-Job-Id"


class TranscriptionEventHub:
    """Broadcast transcription log/progress events to WebSocket subscribers."""

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)

    def emit(self, event: dict[str, Any]) -> None:
        """Schedule broadcast from sync or async context (e.g. thread pool workers)."""
        event.setdefault("ts", datetime.now().isoformat(timespec="seconds"))
        loop = self._loop
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                logger.debug("No event loop for transcription event: %s", event.get("type"))
                return
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(self._broadcast(event), loop)

    async def _broadcast(self, event: dict[str, Any]) -> None:
        async with self._lock:
            targets = list(self._connections)
        dead: list[WebSocket] = []
        for ws in targets:
            try:
                await ws.send_json(event)
            except Exception as err:
                logger.debug("WebSocket send failed: %s", err)
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._connections.discard(ws)

    def log(
        self,
        *,
        job_id: str | None,
        level: str,
        msg: str,
        file: str | None = None,
    ) -> None:
        self.emit(
            {
                "type": "log",
                "job_id": job_id,
                "level": level,
                "msg": msg,
                "file": file,
            }
        )

    def progress(
        self,
        *,
        job_id: str | None,
        processed: int,
        total: int,
        current_file: str | None = None,
        progress: float | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "type": "progress",
            "job_id": job_id,
            "processed": processed,
            "total": total,
            "current_file": current_file,
        }
        if progress is not None:
            payload["progress"] = progress
        self.emit(payload)

    def status(self, *, job_id: str | None, is_running: bool, **extra: Any) -> None:
        self.emit({"type": "status", "job_id": job_id, "is_running": is_running, **extra})

    def done(self, *, job_id: str | None, ok: int = 0, failed: int = 0) -> None:
        self.emit({"type": "done", "job_id": job_id, "ok": ok, "failed": failed})


def get_hub(app: Any) -> TranscriptionEventHub:
    """Return the shared hub from FastAPI app state."""
    hub = getattr(app.state, "transcription_events", None)
    if hub is None:
        hub = TranscriptionEventHub()
        app.state.transcription_events = hub
    return hub
