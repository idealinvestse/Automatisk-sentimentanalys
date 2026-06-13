"""Transcription queue service – persistent JSON + async worker.

Fas 3 – docs/MIGRATION_TO_NICEGUI_PLAN.md §3 (Transkriberings Monitor)
Uses nicegui_api_client for /transcribe, /batch_transcribe, /scan_process.
WebSocket push via transcription_ws_client when API mode is active.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CACHE_DIR = Path(".cache")
QUEUE_STATE_FILE = CACHE_DIR / "transcription_queue.json"
SCAN_STATE_FILE = CACHE_DIR / "transcription_scan_state.json"
_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


def _default_status() -> dict[str, Any]:
    return {
        "is_running": False,
        "current_file": None,
        "progress": 0.0,
        "processed": 0,
        "total": 0,
        "start_time": None,
        "use_api": False,
    }


def _default_settings() -> dict[str, Any]:
    return {
        "backend": "faster",
        "model": "KBLab/kb-whisper-large",
        "device": "auto",
        "language": "sv",
        "hotwords": "",
        "preprocess": True,
        "diarize": False,
        "num_speakers": None,
    }


@dataclass
class TranscriptionState:
    """Persistent transcription monitor state (survives dashboard restarts)."""

    pending_folder: str = "inputs/pending"
    output_dir: str = "outputs/transcripts"
    status: dict[str, Any] = field(default_factory=_default_status)
    logs: list[dict[str, Any]] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=_default_settings)
    pause: bool = False
    stop: bool = False
    queue: list[Path] = field(default_factory=list)
    api_client: Any = field(default=None, repr=False)
    api_strategy: str = "transcribe"  # transcribe | scan_process | batch_transcribe
    job_id: str | None = None
    ws_connected: bool = False
    ws_status: str = "disconnected"  # connected | reconnecting | disconnected
    _worker_task: asyncio.Task | None = field(default=None, repr=False)
    _ws_listener: Any = field(default=None, repr=False)
    _listeners: list[Callable[[str, dict[str, Any]], None]] = field(default_factory=list, repr=False)

    def ensure_cache_dir(self) -> None:
        CACHE_DIR.mkdir(exist_ok=True)

    def save(self) -> None:
        self.ensure_cache_dir()
        try:
            data = {
                "pending": [str(p) for p in self.queue],
                "status": self.status,
                "last_updated": datetime.now().isoformat(),
            }
            with open(QUEUE_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError as err:
            logger.warning("Could not save transcription queue: %s", err)

    def load(self) -> None:
        self.ensure_cache_dir()
        try:
            if not QUEUE_STATE_FILE.exists():
                return
            with open(QUEUE_STATE_FILE, encoding="utf-8") as f:
                data = json.load(f)
            if "pending" in data:
                self.queue = [Path(p) for p in data["pending"]]
            if "status" in data and not self.status.get("is_running"):
                self.status.update(data.get("status", {}))
        except (OSError, json.JSONDecodeError) as err:
            logger.warning("Could not load transcription queue: %s", err)

    def add_listener(self, callback: Callable[[str, dict[str, Any]], None]) -> None:
        """Register UI callback for push updates (event_type, payload)."""
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[str, dict[str, Any]], None]) -> None:
        with contextlib.suppress(ValueError):
            self._listeners.remove(callback)

    def _notify(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        data = payload or {}
        for cb in list(self._listeners):
            try:
                cb(event_type, data)
            except Exception as err:
                logger.debug("Listener error: %s", err)

    def add_log(
        self,
        level: str,
        msg: str,
        file: str | None = None,
        *,
        source: str = "local",
        notify: bool = True,
    ) -> None:
        entry = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "level": level,
            "msg": msg,
            "file": file,
            "source": source,
        }
        self.logs.append(entry)
        if len(self.logs) > 200:
            self.logs = self.logs[-150:]
        self.save()
        if notify:
            self._notify("log", entry)

    def apply_ws_event(self, event: dict[str, Any]) -> None:
        """Merge server WebSocket event into local state."""
        event_type = event.get("type")
        if event_type == "log":
            self.add_log(
                event.get("level", "INFO"),
                event.get("msg", ""),
                file=event.get("file"),
                source="ws",
            )
        elif event_type == "progress":
            self.status["processed"] = event.get("processed", self.status.get("processed", 0))
            self.status["total"] = event.get("total", self.status.get("total", 0))
            if event.get("current_file") is not None:
                self.status["current_file"] = event.get("current_file")
            if event.get("progress") is not None:
                self.status["progress"] = event.get("progress")
            self.save()
            self._notify("progress", dict(self.status))
        elif event_type == "status":
            self.status["is_running"] = event.get("is_running", self.status.get("is_running"))
            for key in ("total", "processed", "current_file", "progress"):
                if key in event:
                    self.status[key] = event[key]
            self.save()
            self._notify("status", dict(self.status))
        elif event_type == "done":
            self._notify("done", event)
        elif event_type == "connected":
            self._set_ws_status("connected")

    def scan_pending(self) -> list[Path]:
        folder = Path(self.pending_folder)
        folder.mkdir(parents=True, exist_ok=True)
        files = sorted(
            [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in _AUDIO_EXTS],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        existing = {str(f) for f in self.queue}
        for f in files:
            if str(f) not in existing:
                self.queue.append(f)
        self.save()
        return self.queue

    def clear_all(self) -> None:
        self.queue.clear()
        self.logs.clear()
        self.status = _default_status()
        self.pause = False
        self.stop = False
        if QUEUE_STATE_FILE.exists():
            QUEUE_STATE_FILE.unlink(missing_ok=True)

    def _api_settings(self) -> dict[str, Any]:
        return dict(self.settings)

    def _api_logs_via_ws(self) -> bool:
        """True when API batch is tagged with a job id (server streams logs via WS)."""
        return bool(self.status.get("use_api") and self.api_client and self.job_id)

    def _set_ws_status(self, status: str) -> None:
        self.ws_status = status
        self.ws_connected = status == "connected"
        self._notify("ws", {"status": status, "connected": self.ws_connected})

    def _on_ws_status_change(self, status: str) -> None:
        self._set_ws_status(status)

    async def _start_ws_listener(self) -> None:
        if not (self.status.get("use_api") and self.api_client):
            return
        from app.nicegui_dashboard.services.transcription_ws_client import TranscriptionWSListener

        self.job_id = str(uuid.uuid4())
        self.api_client.set_job_id(self.job_id)
        if self._ws_listener is None:
            self._ws_listener = TranscriptionWSListener(
                self.api_client,
                on_event=self.apply_ws_event,
                on_status_change=self._on_ws_status_change,
                max_attempts=0,
            )
        await self._ws_listener.start(self.job_id)
        if self.ws_connected:
            self.add_log("INFO", f"WebSocket ansluten (job {self.job_id[:8]}…)", source="local")

    async def _stop_ws_listener(self) -> None:
        if self._ws_listener:
            await self._ws_listener.stop()
        self._set_ws_status("disconnected")
        if self.api_client:
            self.api_client.set_job_id(None)
        self.job_id = None

    async def request_ws_reconnect(self) -> None:
        """Manual WebSocket reconnect (e.g. from UI button)."""
        if not self.api_client or not self.status.get("use_api"):
            self.add_log("WARNING", "WebSocket reconnect kräver API-läge", source="local")
            return
        if not self.job_id:
            self.job_id = str(uuid.uuid4())
            self.api_client.set_job_id(self.job_id)
        if self._ws_listener is None:
            from app.nicegui_dashboard.services.transcription_ws_client import TranscriptionWSListener

            self._ws_listener = TranscriptionWSListener(
                self.api_client,
                on_event=self.apply_ws_event,
                on_status_change=self._on_ws_status_change,
                max_attempts=0,
            )
        self.add_log("INFO", "WebSocket reconnect begärd…", source="local")
        await self._ws_listener.reconnect_now(self.job_id)

    def needs_polling_fallback(self) -> bool:
        """True when API batch runs but WebSocket is not connected."""
        return bool(
            self.status.get("use_api")
            and self.status.get("is_running")
            and self.ws_status != "connected"
        )

    async def _ensure_api_ready(self) -> bool:
        if not self.api_client:
            self.add_log("ERROR", "API-klient saknas")
            return False
        ready = await self.api_client.wait_for_health(attempts=5, interval=1.0)
        if not ready:
            self.add_log("ERROR", f"Backend ej tillgänglig ({self.api_client.base_url})")
        return ready

    async def _process_file_api(self, fpath: Path) -> None:
        """POST /transcribe for a single queued file."""
        from app.nicegui_dashboard.services.nicegui_api_client import APIError

        if not self.api_client:
            raise APIError("API-klient saknas")

        if not self._api_logs_via_ws():
            self.add_log("INFO", f"[API] Transkriberar {fpath.name}...", file=fpath.name)
        result = await self.api_client.transcribe(str(fpath), **self._api_settings())
        if not self._api_logs_via_ws():
            transcript = result.get("transcript") or {}
            n_seg = len(transcript.get("segments") or [])
            self.add_log("INFO", f"[API] Klart: {fpath.name} – {n_seg} segment", file=fpath.name)

    async def _run_batch_scan_process(self) -> list[str]:
        """POST /scan_process on pending folder with persistent state_file."""
        from app.nicegui_dashboard.services.nicegui_api_client import APIError

        if not self.api_client:
            raise APIError("API-klient saknas")

        self.ensure_cache_dir()
        if not self._api_logs_via_ws():
            self.add_log("INFO", f"[API] scan_process på {self.pending_folder}")
        result = await self.api_client.scan_process(
            self.pending_folder,
            state_file=str(SCAN_STATE_FILE),
            operation="transcribe",
            batch_size=4,
            **self._api_settings(),
        )
        processed: list[str] = []
        if not self._api_logs_via_ws():
            for item in result.get("items") or []:
                fname = item.get("file", "?")
                if item.get("ok"):
                    self.add_log(
                        "INFO",
                        f"[scan_process] Klart: {Path(fname).name}",
                        file=Path(fname).name,
                    )
                elif item.get("error"):
                    self.add_log("ERROR", f"[scan_process] {item['error']}", file=Path(fname).name)
            skipped = result.get("skipped", 0)
            if skipped:
                self.add_log("INFO", f"[scan_process] Hoppade över {skipped} redan bearbetade filer")
        else:
            for item in result.get("items") or []:
                if item.get("ok"):
                    processed.append(item.get("file", ""))
        return processed

    async def _run_batch_api_transcribe(self, files: list[Path]) -> list[str]:
        """POST /batch_transcribe for explicit file list."""
        from app.nicegui_dashboard.services.nicegui_api_client import APIError

        if not self.api_client:
            raise APIError("API-klient saknas")

        paths = [str(f) for f in files]
        if not self._api_logs_via_ws():
            self.add_log("INFO", f"[API] batch_transcribe på {len(paths)} filer")
        result = await self.api_client.batch_transcribe(paths, **self._api_settings())
        processed: list[str] = []
        for item in result.get("items") or []:
            fname = item.get("file", "?")
            if item.get("transcript") and not item.get("error"):
                if not self._api_logs_via_ws():
                    n_seg = len((item.get("transcript") or {}).get("segments") or [])
                    self.add_log(
                        "INFO",
                        f"[batch] Klart: {Path(fname).name} – {n_seg} seg",
                        file=Path(fname).name,
                    )
                processed.append(fname)
            elif item.get("error") and not self._api_logs_via_ws():
                self.add_log("ERROR", f"[batch] {item['error']}", file=Path(fname).name)
        return processed

    async def _process_file_local(self, fpath: Path) -> None:
        try:
            from src.transcription.factory import get_transcriber

            transcriber = get_transcriber(
                backend=self.settings.get("backend", "faster"),
                model_name=self.settings.get("model", "KBLab/kb-whisper-large"),
                device=self.settings.get("device", "auto"),
            )
            tr_obj = transcriber.transcribe(
                audio_path=str(fpath),
                language=self.settings.get("language", "sv"),
                preprocess=self.settings.get("preprocess", True),
                diarize=self.settings.get("diarize", False),
            )
            tr = tr_obj.to_dict() if hasattr(tr_obj, "to_dict") else {}
            self.add_log(
                "INFO",
                f"Klart: {fpath.name} – {len(tr.get('segments', []))} segment",
                file=fpath.name,
            )
        except Exception as err:
            self.add_log("ERROR", f"Kunde inte transkribera: {err}. Simulerar.", file=fpath.name)
            await asyncio.sleep(1.2)
            self.add_log("INFO", f"[SIM] Klart: {fpath.name}", file=fpath.name)

    async def run_batch(self, files: list[Path]) -> None:
        """Async batch worker – non-blocking for NiceGUI event loop."""
        self.add_log("INFO", f"Startar batch på {len(files)} filer")
        self.status["is_running"] = True
        self.status["total"] = len(files)
        self.status["processed"] = 0
        self.status["start_time"] = datetime.now().isoformat()
        self.save()
        self._notify("status", dict(self.status))

        use_api = self.status.get("use_api", False)
        processed_paths: list[str] = []

        try:
            if use_api:
                await self._start_ws_listener()
            if use_api:
                if not await self._ensure_api_ready():
                    self.add_log("WARNING", "Faller tillbaka till lokal transkribering")
                    use_api = False
                    self.status["use_api"] = False

            if use_api and self.api_strategy == "scan_process":
                self.status["current_file"] = "scan_process..."
                processed_paths = await self._run_batch_scan_process()
                self.status["processed"] = len(processed_paths)
                self.status["progress"] = 1.0
                self.save()
            elif use_api and self.api_strategy == "batch_transcribe" and not self.stop:
                processed_paths = await self._run_batch_api_transcribe(files)
                self.status["processed"] = len(processed_paths)
                self.status["progress"] = 1.0
                self.save()
            else:
                for i, fpath in enumerate(files):
                    if self.stop:
                        break
                    while self.pause and not self.stop:
                        await asyncio.sleep(0.3)

                    self.status["current_file"] = fpath.name
                    self.status["progress"] = round(i / max(1, len(files)), 2)
                    self.add_log("INFO", f"Bearbetar {fpath.name}...", file=fpath.name)

                    try:
                        if use_api:
                            await self._process_file_api(fpath)
                        else:
                            await self._process_file_local(fpath)
                    except Exception as err:
                        self.add_log("ERROR", f"Fel på {fpath.name}: {err}", file=fpath.name)

                    self.status["processed"] = i + 1
                    self.status["progress"] = round((i + 1) / max(1, len(files)), 2)
                    processed_paths.append(str(fpath))
                    self.save()
        finally:
            await self._stop_ws_listener()
            self.status["is_running"] = False
            self.status["current_file"] = None
            if processed_paths:
                done_set = set(processed_paths)
                self.queue = [f for f in self.queue if str(f) not in done_set]
            self.pause = False
            self.stop = False
            self.save()
            self.add_log("INFO", "Batch slutförd")
            self._notify("status", dict(self.status))

    def start_batch(self) -> bool:
        """Start async batch if queue non-empty and not already running."""
        if self.status.get("is_running"):
            return False
        pending = self.scan_pending()
        if not pending:
            return False
        self.stop = False
        self.pause = False
        files = list(pending)
        self._worker_task = asyncio.create_task(self.run_batch(files))
        return True

    def request_pause(self) -> None:
        self.pause = True
        self.add_log("WARNING", "Paus begärd")

    def request_resume(self) -> None:
        self.pause = False
        self.add_log("INFO", "Återupptagen")

    def request_stop(self) -> None:
        self.stop = True
        self.add_log("WARNING", "Stopp begärd")


def create_transcription_state(api_client: Any = None) -> TranscriptionState:
    """Factory: load persistent state from disk."""
    state = TranscriptionState(api_client=api_client)
    state.load()
    state.scan_pending()
    return state