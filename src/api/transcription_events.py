"""Real-time transcription event hub for WebSocket clients.

Fas 3 WebSocket event hub (in-process or Redis pub/sub).
Thread-safe emit from batch workers; async broadcast to subscribers.

When Redis is available (``API_USE_REDIS_CACHE``), events are published on a
shared channel so all uvicorn workers deliver live logs to subscribers.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)

JOB_HEADER = "X-Transcription-Job-Id"
_REDIS_CHANNEL = "ws:transcription:events"


@dataclass
class _WsConnection:
    websocket: WebSocket
    job_id: str | None = None


class TranscriptionEventHub:
    """Broadcast transcription log/progress events to WebSocket subscribers."""

    def __init__(self, *, redis_client: Any | None = None) -> None:
        self._connections: list[_WsConnection] = []
        self._lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._redis = redis_client
        self._listener_task: asyncio.Task[None] | None = None

    @property
    def backend(self) -> str:
        return "redis" if self._redis is not None else "memory"

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def start_redis_listener(self) -> None:
        """Subscribe to cross-worker events (no-op without Redis)."""
        if self._redis is None or self._listener_task is not None:
            return

        async def _listen() -> None:
            redis = self._redis
            if redis is None:
                return
            pubsub = redis.pubsub(ignore_subscribe_messages=True)
            try:
                pubsub.subscribe(_REDIS_CHANNEL)
                while True:
                    msg = await asyncio.to_thread(pubsub.get_message, timeout=1.0)
                    if not msg or msg.get("type") != "message":
                        await asyncio.sleep(0.05)
                        continue
                    raw = msg.get("data")
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8")
                    try:
                        event = json.loads(raw)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    await self._broadcast(event, from_redis=True)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Redis transcription listener stopped: %s", exc)
            finally:
                try:
                    pubsub.unsubscribe(_REDIS_CHANNEL)
                    pubsub.close()
                except Exception:
                    pass

        self._listener_task = asyncio.create_task(_listen())

    async def stop_redis_listener(self) -> None:
        if self._listener_task is not None:
            self._listener_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._listener_task
            self._listener_task = None

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.append(_WsConnection(websocket=websocket))

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections = [c for c in self._connections if c.websocket is not websocket]

    async def set_subscription(self, websocket: WebSocket, job_id: str | None) -> None:
        async with self._lock:
            for conn in self._connections:
                if conn.websocket is websocket:
                    conn.job_id = job_id
                    return

    def emit(self, event: dict[str, Any]) -> None:
        """Schedule broadcast from sync or async context (e.g. thread pool workers)."""
        event.setdefault("ts", datetime.now().isoformat(timespec="seconds"))
        self._forward_to_status(event)

        if self._redis is not None:
            try:
                self._redis.publish(_REDIS_CHANNEL, json.dumps(event, default=str))
            except Exception as exc:
                logger.warning("Redis publish failed, local-only broadcast: %s", exc)
                self._schedule_local_broadcast(event)
            return

        self._schedule_local_broadcast(event)

    def _schedule_local_broadcast(self, event: dict[str, Any]) -> None:
        loop = self._loop
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                logger.debug("No event loop for transcription event: %s", event.get("type"))
                return
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(self._broadcast(event), loop)

    def _should_send(self, conn: _WsConnection, event: dict[str, Any]) -> bool:
        event_job = event.get("job_id")
        if not conn.job_id:
            return True
        if not event_job:
            return event.get("type") in ("connected", "pong", "subscribed")
        return bool(event_job == conn.job_id)

    async def _broadcast(self, event: dict[str, Any], *, from_redis: bool = False) -> None:
        async with self._lock:
            targets = list(self._connections)
        dead: list[WebSocket] = []
        for conn in targets:
            if not self._should_send(conn, event):
                continue
            try:
                await conn.websocket.send_json(event)
            except Exception as err:
                logger.debug("WebSocket send failed: %s", err)
                dead.append(conn.websocket)
        if dead:
            async with self._lock:
                self._connections = [c for c in self._connections if c.websocket not in dead]

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

    def _forward_to_status(self, event: dict[str, Any]) -> None:
        """Mirror transcription events into the global StatusReporter."""
        try:
            from ..core.status import get_status_reporter

            reporter = get_status_reporter()
            event_type = str(event.get("type", "info"))
            component = "transcription"
            phase = event_type
            job_id = event.get("job_id")
            if event_type == "log":
                level = str(event.get("level", "INFO")).upper()
                message = str(event.get("msg", ""))
                if level == "ERROR":
                    reporter.error(component, phase, message, job_id=job_id)
                elif level == "WARN":
                    reporter.warn(component, phase, message, job_id=job_id)
                else:
                    reporter.info(component, phase, message, job_id=job_id)
            elif event_type == "progress":
                reporter.progress(
                    component,
                    phase,
                    int(event.get("processed", 0)),
                    int(event.get("total", 1)),
                    message=event.get("current_file"),
                    job_id=job_id,
                )
            elif event_type == "status":
                running = event.get("is_running", False)
                reporter.phase(
                    component,
                    phase,
                    "Transkribering pågår" if running else "Transkribering stoppad",
                    job_id=job_id,
                )
            elif event_type == "done":
                reporter.phase(
                    component,
                    phase,
                    f"Transkribering klar (ok={event.get('ok', 0)}, failed={event.get('failed', 0)})",
                    job_id=job_id,
                )
        except Exception:
            logger.debug("Status forward failed for transcription event", exc_info=True)


def get_hub(app: Any) -> TranscriptionEventHub:
    """Return the shared hub from FastAPI app state."""
    hub = getattr(app.state, "transcription_events", None)
    if hub is None:
        redis_client = None
        cache = getattr(app.state, "cache", None)
        if cache is not None:
            redis_client = getattr(cache, "redis_client", None)
        hub = TranscriptionEventHub(redis_client=redis_client)
        app.state.transcription_events = hub
    return hub
