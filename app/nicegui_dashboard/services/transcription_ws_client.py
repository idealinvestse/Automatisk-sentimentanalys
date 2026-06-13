"""WebSocket client for real-time transcription logs from FastAPI backend.

Fas 3 WebSocket – docs/MIGRATION_TO_NICEGUI_PLAN.md §3
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlencode, urlparse, urlunparse

from app.nicegui_dashboard.services.nicegui_api_client import NiceGUIAPIClient

logger = logging.getLogger(__name__)

EventHandler = Callable[[dict[str, Any]], Awaitable[None] | None]


class TranscriptionWSListener:
    """Subscribe to /ws/transcription and forward events to a handler."""

    def __init__(
        self,
        api_client: NiceGUIAPIClient,
        *,
        on_event: EventHandler,
    ) -> None:
        self._client = api_client
        self._on_event = on_event
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self.connected = False
        self.job_id: str | None = None

    def _ws_url(self) -> str:
        parsed = urlparse(self._client.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        query: dict[str, str] = {}
        if self._client.api_key:
            query["api_key"] = self._client.api_key
        netloc = parsed.netloc or parsed.path
        path = (parsed.path.rstrip("/") if parsed.scheme else "") + "/ws/transcription"
        if not path.startswith("/"):
            path = "/" + path
        url = urlunparse((scheme, netloc, path, "", urlencode(query), ""))
        return url

    async def start(self, job_id: str) -> None:
        """Connect and listen until stop() or connection error."""
        await self.stop()
        self.job_id = job_id
        self._stop.clear()
        self._task = asyncio.create_task(self._listen_loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        self.connected = False

    async def _listen_loop(self) -> None:
        try:
            import websockets
        except ImportError:
            logger.warning("websockets saknas – installera uvicorn[standard] eller websockets")
            return

        url = self._ws_url()
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                self.connected = True
                if self.job_id:
                    await ws.send(json.dumps({"type": "subscribe", "job_id": self.job_id}))

                while not self._stop.is_set():
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    except Exception:
                        break
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    await self._dispatch(event)
        except Exception as err:
            logger.info("Transcription WebSocket avslutad: %s", err)
        finally:
            self.connected = False

    async def _dispatch(self, event: dict[str, Any]) -> None:
        job_id = event.get("job_id")
        if self.job_id and job_id and job_id != self.job_id:
            return
        result = self._on_event(event)
        if asyncio.iscoroutine(result):
            await result