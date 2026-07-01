"""WebSocket client for real-time transcription logs from FastAPI backend.

Fas 6.1 – docs/archive/MIGRATION_TO_NICEGUI_PLAN.md (WebSocket reconnect)
Exponential backoff, max attempts, status callbacks.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlparse, urlunparse

from app.archive.nicegui_dashboard.services.nicegui_api_client import NiceGUIAPIClient

logger = logging.getLogger(__name__)

EventHandler = Callable[[dict[str, Any]], Awaitable[None] | None]
StatusHandler = Callable[[str], None]

WS_CONNECTED = "connected"
WS_RECONNECTING = "reconnecting"
WS_DISCONNECTED = "disconnected"


class TranscriptionWSListener:
    """Subscribe to /ws/transcription with automatic reconnect."""

    def __init__(
        self,
        api_client: NiceGUIAPIClient,
        *,
        on_event: EventHandler,
        on_status_change: StatusHandler | None = None,
        max_attempts: int = 0,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
    ) -> None:
        self._client = api_client
        self._on_event = on_event
        self._on_status_change = on_status_change
        self._max_attempts = max_attempts  # 0 = unlimited while running
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._reconnect_requested = asyncio.Event()
        self.connected = False
        self.status = WS_DISCONNECTED
        self.job_id: str | None = None
        self._attempt = 0

    def _set_status(self, status: str) -> None:
        if self.status == status:
            return
        self.status = status
        self.connected = status == WS_CONNECTED
        if self._on_status_change:
            self._on_status_change(status)

    def _ws_url(self) -> str:
        parsed = urlparse(self._client.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        netloc = parsed.netloc or parsed.path
        path = (parsed.path.rstrip("/") if parsed.scheme else "") + "/ws/transcription"
        if not path.startswith("/"):
            path = "/" + path
        return urlunparse((scheme, netloc, path, "", "", ""))

    def _ws_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._client.api_key:
            headers["X-API-Key"] = self._client.api_key
        return headers

    async def start(self, job_id: str) -> None:
        """Connect and listen until stop() or max attempts exhausted."""
        await self.stop()
        self.job_id = job_id
        self._stop.clear()
        self._attempt = 0
        self._task = asyncio.create_task(self._listen_loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        self.connected = False
        self._set_status(WS_DISCONNECTED)

    async def reconnect_now(self, job_id: str | None = None) -> None:
        """Manual reconnect – reset backoff and retry immediately."""
        if job_id:
            self.job_id = job_id
        self._attempt = 0
        self._reconnect_requested.set()
        if self._task is None or self._task.done():
            self._stop.clear()
            self._task = asyncio.create_task(self._listen_loop())

    async def _listen_loop(self) -> None:
        try:
            import websockets
        except ImportError:
            logger.warning("websockets saknas – installera uvicorn[standard] eller websockets")
            self._set_status(WS_DISCONNECTED)
            return

        url = self._ws_url()
        while not self._stop.is_set():
            if self._max_attempts > 0 and self._attempt >= self._max_attempts:
                logger.info("WebSocket max attempts (%s) nådd", self._max_attempts)
                self._set_status(WS_DISCONNECTED)
                break

            if self._attempt > 0:
                self._set_status(WS_RECONNECTING)
            else:
                self._set_status(WS_DISCONNECTED)

            try:
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=20,
                    additional_headers=self._ws_headers(),
                ) as ws:
                    self._set_status(WS_CONNECTED)
                    self._attempt = 0
                    if self.job_id:
                        await ws.send(json.dumps({"type": "subscribe", "job_id": self.job_id}))

                    while not self._stop.is_set():
                        if self._reconnect_requested.is_set():
                            self._reconnect_requested.clear()
                            break
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        except TimeoutError:
                            continue
                        except Exception:
                            break
                        try:
                            event = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        await self._dispatch(event)
            except asyncio.CancelledError:
                raise
            except Exception as err:
                logger.info("Transcription WebSocket fel: %s", err)
            finally:
                if not self._stop.is_set():
                    self.connected = False
                    if self.status == WS_CONNECTED:
                        self._set_status(WS_RECONNECTING)

            if self._stop.is_set():
                break

            self._attempt += 1
            if self._max_attempts > 0 and self._attempt >= self._max_attempts:
                self._set_status(WS_DISCONNECTED)
                break

            delay = min(self._max_delay, self._base_delay * (2 ** max(0, self._attempt - 1)))
            self._set_status(WS_RECONNECTING)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
                break
            except TimeoutError:
                continue

        self.connected = False
        if self.status != WS_DISCONNECTED:
            self._set_status(WS_DISCONNECTED)

    async def _dispatch(self, event: dict[str, Any]) -> None:
        job_id = event.get("job_id")
        if self.job_id and job_id and job_id != self.job_id:
            return
        result = self._on_event(event)
        if asyncio.iscoroutine(result):
            await result
