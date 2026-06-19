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

from app.nicegui_dashboard.services.transcription_runtime import (
    api_retry_count,
    build_local_transcribe_kwargs,
    local_timeout_seconds,
    resolve_hotwords,
    validate_audio_file,
    validate_upload_bytes,
)

logger = logging.getLogger(__name__)

CACHE_DIR = Path(".cache")
QUEUE_STATE_FILE = CACHE_DIR / "transcription_queue.json"
SCAN_STATE_FILE = CACHE_DIR / "transcription_scan_state.json"
ADHOC_UPLOAD_DIR = CACHE_DIR / "adhoc_uploads"
_STATE_VERSION = 3
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
        "active_job_id": None,
        "api_strategy": "transcribe",
    }


def _default_settings() -> dict[str, Any]:
    return {
        "backend": "faster",
        "model": "kb-whisper-large",
        "device": "auto",
        "language": "sv",
        "hotwords": "",
        "preprocess": True,
        "diarize": False,
        "num_speakers": None,
        "beam_size": 5,
        "vad": True,
        "chunk_length_s": 30,
        "revision": None,
        "initial_prompt": None,
        "word_timestamps": True,
        "workers": 1,
        "worker_timeout": 300.0,
        "sentiment_profile": "callcenter",
        "use_hotwords_file": True,
        "hotwords_file": "configs/callcenter_hotwords.txt",
        "local_timeout_s": 900.0,
        "api_retries": 2,
        "api_fallback_local": True,
    }


def _default_scan_config() -> dict[str, Any]:
    return {
        "operation": "transcribe",
        "batch_size": 4,
        "pattern": None,
        "max_files": None,
        "recursive": True,
    }


@dataclass
class TranscriptionState:
    """Persistent transcription monitor state (survives dashboard restarts)."""

    pending_folder: str = "inputs/pending"
    output_dir: str = "outputs/transcripts"
    status: dict[str, Any] = field(default_factory=_default_status)
    logs: list[dict[str, Any]] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=_default_settings)
    scan_config: dict[str, Any] = field(default_factory=_default_scan_config)
    active_preset: str = "callcenter_standard"
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
    # Ad-hoc single-file transcription (separate from batch queue)
    adhoc_upload_path: Path | None = None
    adhoc_filename: str | None = None
    adhoc_running: bool = False
    adhoc_progress: float = 0.0
    adhoc_status_text: str = ""
    adhoc_result: dict[str, Any] | None = None
    adhoc_meta: dict[str, Any] | None = None
    adhoc_error: str | None = None
    adhoc_cancel: bool = False
    _adhoc_task: asyncio.Task | None = field(default=None, repr=False)

    def ensure_cache_dir(self) -> None:
        CACHE_DIR.mkdir(exist_ok=True)

    def update_setting(self, key: str, value: Any, *, section: str = "settings") -> None:
        """Update a setting/scan_config field and persist."""
        target = self.scan_config if section == "scan" else self.settings
        target[key] = value
        if section == "settings" and key in ("api_strategy",):
            self.api_strategy = str(value)
        self.active_preset = "anpassad"
        self.save()

    def save(self) -> None:
        self.ensure_cache_dir()
        try:
            data = {
                "version": _STATE_VERSION,
                "pending": [str(p) for p in self.queue],
                "status": self.status,
                "settings": self.settings,
                "scan_config": self.scan_config,
                "api_strategy": self.api_strategy,
                "pending_folder": self.pending_folder,
                "output_dir": self.output_dir,
                "active_preset": self.active_preset,
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
            version = data.get("version", 1)
            if version >= 2:
                self.settings.update(data.get("settings", {}))
                self.scan_config.update(data.get("scan_config", {}))
                self.api_strategy = data.get("api_strategy", self.api_strategy)
                self.pending_folder = data.get("pending_folder", self.pending_folder)
                self.output_dir = data.get("output_dir", self.output_dir)
                self.active_preset = data.get("active_preset", self.active_preset)
            if "status" in data and not self.status.get("is_running"):
                loaded = data.get("status", {})
                loaded["is_running"] = False
                loaded["active_job_id"] = None
                self.status.update(loaded)
            if version < 3:
                from app.nicegui_dashboard.services.transcription_presets import apply_default_preset

                apply_default_preset(self)
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
        if notify:
            self._notify("log", entry)

    def _event_for_job(self, event: dict[str, Any]) -> bool:
        event_job = event.get("job_id")
        if not event_job:
            return True
        active = self.job_id or self.status.get("active_job_id")
        return not active or event_job == active

    def apply_ws_event(self, event: dict[str, Any]) -> None:
        """Merge server WebSocket event into local state."""
        if not self._event_for_job(event):
            return
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
        candidates = sorted(
            [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in _AUDIO_EXTS],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        files: list[Path] = []
        for f in candidates:
            try:
                validate_audio_file(f)
                files.append(f)
            except ValueError as err:
                self.add_log("WARNING", str(err), file=f.name, notify=False)
        existing = {str(f) for f in self.queue}
        for f in files:
            if str(f) not in existing:
                self.queue.append(f)
        self.save()
        return self.queue

    def clear_all(self) -> None:
        self.clear_queue()
        self.clear_logs()
        self.status = _default_status()
        self.pause = False
        self.stop = False
        if QUEUE_STATE_FILE.exists():
            QUEUE_STATE_FILE.unlink(missing_ok=True)

    def clear_queue(self) -> None:
        self.queue.clear()
        self.save()
        self._notify("queue", {})

    def clear_logs(self) -> None:
        self.logs.clear()
        self._notify("log_clear", {})

    def remove_from_queue(self, path_str: str) -> None:
        self.queue = [f for f in self.queue if str(f) != path_str]
        self.save()
        self._notify("queue", {})

    def queue_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for f in self.queue:
            try:
                stat = f.stat()
                rows.append(
                    {
                        "name": f.name,
                        "path": str(f),
                        "size_kb": round(stat.st_size / 1024, 1),
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                    }
                )
            except OSError:
                rows.append({"name": f.name, "path": str(f), "size_kb": 0, "modified": "—"})
        return rows

    def filtered_logs(
        self,
        *,
        level: str | None = None,
        source: str | None = None,
        search: str = "",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for entry in reversed(self.logs):
            if level and entry.get("level") != level:
                continue
            if source and entry.get("source") != source:
                continue
            if search:
                hay = f"{entry.get('msg', '')} {entry.get('file', '')}".lower()
                if search.lower() not in hay:
                    continue
            out.append(entry)
            if len(out) >= limit:
                break
        return list(reversed(out))

    def elapsed_seconds(self) -> float | None:
        start = self.status.get("start_time")
        if not start:
            return None
        try:
            started = datetime.fromisoformat(start)
            return (datetime.now() - started).total_seconds()
        except ValueError:
            return None

    def _api_settings(self) -> dict[str, Any]:
        merged = dict(self.settings)
        merged.update(self.scan_config)
        hotwords = resolve_hotwords(self.settings)
        if hotwords:
            merged["hotwords"] = hotwords
        return merged

    async def _with_retries(
        self,
        operation: str,
        coro_factory: Callable[[], Any],
        *,
        file_name: str | None = None,
    ) -> Any:
        """Run an async API call with configurable retries and backoff."""
        retries = api_retry_count(self.settings)
        last_err: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                return await coro_factory()
            except Exception as err:
                last_err = err
                if attempt >= retries:
                    break
                self.add_log(
                    "WARNING",
                    f"{operation} försök {attempt}/{retries} misslyckades: {err}",
                    file=file_name,
                )
                await asyncio.sleep(min(2.0 * attempt, 6.0))
        assert last_err is not None
        raise last_err

    def _transcribe_file_sync(
        self,
        fpath: Path,
        *,
        on_chunk_progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Blocking local ASR – run via asyncio.to_thread from async code."""
        from src.transcription.factory import get_transcriber

        backend, kwargs = build_local_transcribe_kwargs(
            self.settings,
            fpath,
            on_chunk_progress=on_chunk_progress,
        )
        transcriber = get_transcriber(
            backend=backend,
            model_name=self.settings.get("model", "kb-whisper-large"),
            device=self.settings.get("device", "auto"),
        )
        tr_obj = transcriber.transcribe(**kwargs)
        return tr_obj.to_dict() if hasattr(tr_obj, "to_dict") else dict(tr_obj or {})

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
        self.status["active_job_id"] = self.job_id
        self.api_client.set_job_id(self.job_id)
        if self._ws_listener is None:
            self._ws_listener = TranscriptionWSListener(
                self.api_client,
                on_event=self.apply_ws_event,
                on_status_change=self._on_ws_status_change,
                max_attempts=0,
            )
        await self._ws_listener.start(self.job_id)
        self.save()
        if self.ws_connected:
            self.add_log("INFO", f"WebSocket ansluten (job {self.job_id[:8]}…)", source="local")

    async def _stop_ws_listener(self) -> None:
        if self._ws_listener:
            await self._ws_listener.stop()
        self._set_ws_status("disconnected")
        if self.api_client:
            self.api_client.set_job_id(None)
        self.job_id = None
        self.status["active_job_id"] = None

    async def request_ws_reconnect(self) -> None:
        """Manual WebSocket reconnect (e.g. from UI button)."""
        if not self.api_client or not self.status.get("use_api"):
            self.add_log("WARNING", "WebSocket reconnect kräver API-läge", source="local")
            return
        if not self.job_id:
            self.job_id = str(uuid.uuid4())
            self.status["active_job_id"] = self.job_id
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

    async def _wait_if_paused(self) -> bool:
        """Wait while paused. Returns False if stop was requested."""
        while self.pause and not self.stop:
            await asyncio.sleep(0.3)
        return not self.stop

    async def _process_file_api(self, fpath: Path) -> bool:
        """POST /transcribe for a single queued file. Returns True on success."""
        from app.nicegui_dashboard.services.nicegui_api_client import APIError

        if not self.api_client:
            raise APIError("API-klient saknas")

        validate_audio_file(fpath)

        if not self._api_logs_via_ws():
            self.add_log("INFO", f"[API] Transkriberar {fpath.name}...", file=fpath.name)

        async def _call() -> dict[str, Any]:
            return await self.api_client.transcribe(str(fpath), **self._api_settings())

        try:
            result = await self._with_retries("[API]", _call, file_name=fpath.name)
        except Exception as err:
            if self.settings.get("api_fallback_local", True):
                self.add_log(
                    "WARNING",
                    f"[API] Fel – försöker lokal fallback: {err}",
                    file=fpath.name,
                )
                return await self._process_file_local(fpath)
            self.add_log("ERROR", f"[API] {err}", file=fpath.name)
            return False

        if not self._api_logs_via_ws():
            transcript = result.get("transcript") or {}
            n_seg = len(transcript.get("segments") or [])
            self.add_log("INFO", f"[API] Klart: {fpath.name} – {n_seg} segment", file=fpath.name)
        return True

    async def _run_batch_scan_process(self) -> list[str]:
        """POST /scan_process on pending folder with persistent state_file."""
        from app.nicegui_dashboard.services.nicegui_api_client import APIError

        if not self.api_client:
            raise APIError("API-klient saknas")

        sc = self.scan_config
        operation = sc.get("operation", "transcribe")
        self.ensure_cache_dir()
        if not self._api_logs_via_ws():
            label = "analys" if operation == "analyze_conversation" else "transkribering"
            self.add_log("INFO", f"[API] scan_process ({label}) på {self.pending_folder}")
        async def _call() -> dict[str, Any]:
            return await self.api_client.scan_process(
                self.pending_folder,
                state_file=str(SCAN_STATE_FILE),
                operation=operation,
                batch_size=int(sc.get("batch_size", 4)),
                max_files=sc.get("max_files"),
                pattern=sc.get("pattern"),
                recursive=bool(sc.get("recursive", True)),
                **self._api_settings(),
            )

        result = await self._with_retries("[scan_process]", _call)
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
        if self.stop and self.job_id and self.api_client:
            with contextlib.suppress(Exception):
                await self.api_client.cancel_job(self.job_id)
        return processed

    async def _run_batch_api_transcribe(self, files: list[Path]) -> list[str]:
        """POST /batch_transcribe for explicit file list."""
        from app.nicegui_dashboard.services.nicegui_api_client import APIError

        if not self.api_client:
            raise APIError("API-klient saknas")

        paths = [str(f) for f in files]
        if not self._api_logs_via_ws():
            self.add_log("INFO", f"[API] batch_transcribe på {len(paths)} filer")
        async def _call() -> dict[str, Any]:
            return await self.api_client.batch_transcribe(paths, **self._api_settings())

        result = await self._with_retries("[batch_transcribe]", _call)
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
        if self.stop and self.job_id and self.api_client:
            with contextlib.suppress(Exception):
                await self.api_client.cancel_job(self.job_id)
        return processed

    async def _process_file_local(self, fpath: Path) -> bool:
        """Run local ASR with validation and timeout. Returns True on success."""
        timeout = local_timeout_seconds(self.settings)
        try:
            validate_audio_file(fpath)
            tr = await asyncio.wait_for(
                asyncio.to_thread(self._transcribe_file_sync, fpath),
                timeout=timeout,
            )
            self.add_log(
                "INFO",
                f"Klart: {fpath.name} – {len(tr.get('segments', []))} segment",
                file=fpath.name,
            )
            return True
        except TimeoutError:
            self.add_log(
                "ERROR",
                f"Timeout efter {int(timeout)}s: {fpath.name}",
                file=fpath.name,
            )
            return False
        except Exception as err:
            self.add_log("ERROR", f"Kunde inte transkribera: {err}", file=fpath.name)
            return False

    async def run_batch(self, files: list[Path]) -> None:
        """Async batch worker – non-blocking for NiceGUI event loop."""
        self.add_log("INFO", f"Startar batch på {len(files)} filer (strategi: {self.api_strategy})")
        self.status["is_running"] = True
        self.status["total"] = len(files)
        self.status["processed"] = 0
        self.status["start_time"] = datetime.now().isoformat()
        self.status["api_strategy"] = self.api_strategy
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

            if use_api and self.api_strategy == "scan_process" and not self.stop:
                self.status["current_file"] = "scan_process..."
                processed_paths = await self._run_batch_scan_process()
                self.status["processed"] = len(processed_paths) if processed_paths else self.status.get("processed", 0)
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
                    if not await self._wait_if_paused():
                        break

                    self.status["current_file"] = fpath.name
                    self.status["progress"] = round(i / max(1, len(files)), 2)
                    self.add_log("INFO", f"Bearbetar {fpath.name}...", file=fpath.name)

                    ok = False
                    try:
                        if use_api:
                            ok = await self._process_file_api(fpath)
                        else:
                            ok = await self._process_file_local(fpath)
                    except Exception as err:
                        self.add_log("ERROR", f"Fel på {fpath.name}: {err}", file=fpath.name)

                    self.status["processed"] = i + 1
                    self.status["progress"] = round((i + 1) / max(1, len(files)), 2)
                    if ok:
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
            was_stopped = self.stop
            self.stop = False
            self.save()
            msg = "Batch avbruten" if was_stopped else "Batch slutförd"
            self.add_log("INFO", msg)
            self._notify("status", dict(self.status))

    def start_batch(self) -> bool:
        """Start async batch if queue non-empty and not already running."""
        if self.status.get("is_running"):
            return False
        pending = self.scan_pending()
        if not pending and self.api_strategy != "scan_process":
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

    async def request_cancel(self) -> None:
        """Cancel running backend job and stop local worker."""
        self.stop = True
        self.add_log("WARNING", "Avbryt begärd")
        if self.job_id and self.api_client:
            try:
                await self.api_client.cancel_job(self.job_id)
                self.add_log("INFO", f"Backend cancel skickad (job {self.job_id[:8]}…)")
            except Exception as err:
                self.add_log("WARNING", f"Cancel misslyckades: {err}")

    def _notify_adhoc(self) -> None:
        self._notify(
            "adhoc",
            {
                "running": self.adhoc_running,
                "progress": self.adhoc_progress,
                "status_text": self.adhoc_status_text,
                "filename": self.adhoc_filename,
                "result": self.adhoc_result,
                "meta": self.adhoc_meta,
                "error": self.adhoc_error,
            },
        )

    def save_adhoc_upload(self, content: bytes, filename: str) -> Path:
        """Persist uploaded audio for ad-hoc transcription."""
        suffix = Path(filename).suffix.lower()
        if suffix not in _AUDIO_EXTS:
            raise ValueError(f"Filtyp {suffix or '(saknas)'} stöds inte. Tillåtna: {', '.join(sorted(_AUDIO_EXTS))}")
        validate_upload_bytes(content, filename)

        self.ensure_cache_dir()
        ADHOC_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = Path(filename).name.replace(" ", "_")
        dest = ADHOC_UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{safe_name}"
        dest.write_bytes(content)

        if self.adhoc_upload_path and self.adhoc_upload_path.exists():
            with contextlib.suppress(OSError):
                self.adhoc_upload_path.unlink()

        self.adhoc_upload_path = dest
        self.adhoc_filename = safe_name
        self.adhoc_result = None
        self.adhoc_meta = None
        self.adhoc_error = None
        self._notify_adhoc()
        return dest

    def clear_adhoc_preview(self) -> None:
        """Reset ad-hoc upload and result preview."""
        if self.adhoc_upload_path and self.adhoc_upload_path.exists():
            with contextlib.suppress(OSError):
                self.adhoc_upload_path.unlink()
        self.adhoc_upload_path = None
        self.adhoc_filename = None
        self.adhoc_result = None
        self.adhoc_meta = None
        self.adhoc_error = None
        self.adhoc_status_text = ""
        self.adhoc_progress = 0.0
        self._notify_adhoc()

    def cancel_adhoc(self) -> None:
        """Request cancellation of running ad-hoc transcription."""
        self.adhoc_cancel = True
        if self._adhoc_task and not self._adhoc_task.done():
            self._adhoc_task.cancel()
        self.adhoc_status_text = "Avbryter…"
        self._notify_adhoc()

    def start_adhoc_transcription(self) -> bool:
        """Start async ad-hoc transcription if upload exists and not already running."""
        if self.adhoc_running:
            return False
        if not self.adhoc_upload_path or not self.adhoc_upload_path.exists():
            return False
        self.adhoc_cancel = False
        self.adhoc_error = None
        self.adhoc_result = None
        self.adhoc_meta = None
        self._adhoc_task = asyncio.create_task(self.run_adhoc_transcription())
        return True

    async def run_adhoc_transcription(self) -> None:
        """Transcribe the uploaded ad-hoc file (API or local)."""
        fpath = self.adhoc_upload_path
        if not fpath or not fpath.exists():
            self.adhoc_error = "Ingen uppladdad fil"
            self._notify_adhoc()
            return

        self.adhoc_running = True
        self.adhoc_progress = 0.05
        self.adhoc_status_text = "Förbereder transkribering…"
        self._notify_adhoc()

        validate_audio_file(fpath)
        want_api = bool(self.status.get("use_api") and self.api_client)
        timeout = local_timeout_seconds(self.settings)
        used_api = False

        try:
            if want_api:
                if not await self._ensure_api_ready():
                    if not self.settings.get("api_fallback_local", True):
                        raise RuntimeError("Backend ej tillgänglig")
                    self.add_log(
                        "WARNING",
                        "Backend ej tillgänglig – lokal fallback",
                        file=fpath.name,
                    )
                else:
                    self.adhoc_status_text = "Transkriberar via API…"
                    self.adhoc_progress = 0.2
                    self._notify_adhoc()

                    from app.nicegui_dashboard.services.test_lab_service import resolve_api_audio_path

                    api_path, warning = resolve_api_audio_path(str(fpath.resolve()))
                    if warning:
                        self.add_log("WARNING", warning, file=fpath.name)

                    async def _call() -> dict[str, Any]:
                        return await self.api_client.transcribe(str(api_path), **self._api_settings())

                    try:
                        result = await self._with_retries("[Ad-hoc API]", _call, file_name=fpath.name)
                    except Exception as err:
                        if not self.settings.get("api_fallback_local", True):
                            raise
                        self.add_log(
                            "WARNING",
                            f"[Ad-hoc] API fel – lokal fallback: {err}",
                            file=fpath.name,
                        )
                    else:
                        if self.adhoc_cancel:
                            return
                        transcript = result.get("transcript") or {}
                        self.adhoc_result = transcript
                        self.adhoc_meta = {"source": "api", "api_path": api_path}
                        used_api = True

            if not used_api:
                self.adhoc_status_text = "Transkriberar lokalt…"
                self.adhoc_progress = 0.15
                self._notify_adhoc()

                def on_chunk(done: int, total: int) -> None:
                    pct = done / max(1, total)
                    self.adhoc_progress = 0.15 + 0.8 * pct
                    self.adhoc_status_text = f"Chunk {done}/{total} – {int(pct * 100)} %"
                    self._notify_adhoc()

                transcript = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._transcribe_file_sync,
                        fpath,
                        on_chunk_progress=on_chunk,
                    ),
                    timeout=timeout,
                )
                if self.adhoc_cancel:
                    return
                self.adhoc_result = transcript
                self.adhoc_meta = {"source": "local"}

            self.adhoc_progress = 1.0
            n_seg = len((self.adhoc_result or {}).get("segments") or [])
            self.adhoc_status_text = f"Klart – {n_seg} segment"
            self.add_log("INFO", f"[Ad-hoc] Klart: {fpath.name} – {n_seg} segment", file=fpath.name)
        except asyncio.CancelledError:
            self.adhoc_status_text = "Avbruten"
            self.add_log("WARNING", f"[Ad-hoc] Avbruten: {fpath.name}", file=fpath.name)
        except TimeoutError:
            self.adhoc_error = f"Timeout efter {int(timeout)}s"
            self.adhoc_status_text = "Fel vid transkribering"
            self.add_log("ERROR", f"[Ad-hoc] {self.adhoc_error}", file=fpath.name)
        except Exception as err:
            self.adhoc_error = str(err)
            self.adhoc_status_text = "Fel vid transkribering"
            self.add_log("ERROR", f"[Ad-hoc] {err}", file=fpath.name)
        finally:
            self.adhoc_running = False
            self.adhoc_cancel = False
            self._notify_adhoc()


def create_transcription_state(api_client: Any = None) -> TranscriptionState:
    """Factory: load persistent state from disk; apply default preset on first run."""
    from app.nicegui_dashboard.services.transcription_presets import apply_default_preset

    state = TranscriptionState(api_client=api_client)
    fresh = not QUEUE_STATE_FILE.exists()
    state.load()
    if fresh:
        apply_default_preset(state)
    state.scan_pending()
    return state