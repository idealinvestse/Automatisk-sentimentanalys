"""httpx async client for the FastAPI backend.

Fas 3 – docs/MIGRATION_TO_NICEGUI_PLAN.md §3
Wraps /health, /analyze_pipeline, /transcribe, /batch_transcribe, /scan_process.
WebSocket logs via /ws/transcription (see transcription_ws_client.py).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 300.0
JOB_HEADER = "X-Transcription-Job-Id"


class APIError(Exception):
    """Raised when the backend returns a non-success status."""

    def __init__(self, message: str, *, status_code: int | None = None, detail: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


class NiceGUIAPIClient:
    """Async HTTP client for the sentiment analysis REST API."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        api_key: str | None = None,
        openrouter_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.openrouter_key = openrouter_key
        self.timeout = timeout
        self.job_id: str | None = None

    @classmethod
    def from_env(cls) -> NiceGUIAPIClient:
        """Create client from environment variables."""
        return cls(
            base_url=os.environ.get("SENTIMENT_API_BASE_URL", DEFAULT_BASE_URL),
            api_key=os.environ.get("SENTIMENT_API_KEY"),
            openrouter_key=os.environ.get("OPENROUTER_API_KEY"),
            timeout=float(os.environ.get("SENTIMENT_API_TIMEOUT", str(DEFAULT_TIMEOUT))),
        )

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        if self.openrouter_key:
            headers["X-OpenRouter-Key"] = self.openrouter_key
        if self.job_id:
            headers[JOB_HEADER] = self.job_id
        return headers

    def set_job_id(self, job_id: str | None) -> None:
        """Attach job id so backend emits matching WebSocket events."""
        self.job_id = job_id

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload, headers=self._headers())
        except httpx.ConnectError as err:
            raise APIError(f"Kan inte ansluta till backend ({self.base_url}): {err}") from err
        except httpx.TimeoutException as err:
            raise APIError(f"Timeout mot {path} ({self.timeout}s)") from err

        if response.status_code >= 400:
            detail: Any
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise APIError(
                f"API-fel {response.status_code} på {path}",
                status_code=response.status_code,
                detail=detail,
            )
        return response.json()

    async def health(self) -> bool:
        """Return True if backend responds OK on /health."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def wait_for_health(self, *, attempts: int = 5, interval: float = 1.0) -> bool:
        """Poll /health until available or attempts exhausted."""
        import asyncio

        for i in range(attempts):
            if await self.health():
                return True
            if i < attempts - 1:
                await asyncio.sleep(interval)
        return False

    async def analyze_pipeline(
        self,
        segments: list[dict[str, Any]],
        *,
        use_mistral_llm: bool = False,
        deep_analysis: bool = False,
        device: str = "auto",
        llm_model: str | None = None,
    ) -> dict[str, Any]:
        """POST /analyze_pipeline – full CallAnalysisPipeline on segments."""
        payload: dict[str, Any] = {
            "segments": segments,
            "device": device,
            "use_mistral_llm": use_mistral_llm,
            "deep_analysis": deep_analysis,
        }
        if llm_model:
            payload["llm_model"] = llm_model
        return await self._post("/analyze_pipeline", payload)

    async def transcribe(self, audio_path: str, **settings: Any) -> dict[str, Any]:
        """POST /transcribe – single-file ASR."""
        payload: dict[str, Any] = {
            "audio_path": audio_path,
            "model": settings.get("model", "kb-whisper-large"),
            "backend": settings.get("backend", "faster"),
            "device": settings.get("device", "auto"),
            "language": settings.get("language", "sv"),
            "diarize": settings.get("diarize", False),
            "preprocess": settings.get("preprocess", False),
        }
        if settings.get("num_speakers") is not None:
            payload["num_speakers"] = settings["num_speakers"]
        hotwords = settings.get("hotwords")
        if hotwords:
            if isinstance(hotwords, str):
                payload["hotwords"] = [w.strip() for w in hotwords.split(",") if w.strip()]
            else:
                payload["hotwords"] = hotwords
        return await self._post("/transcribe", payload)

    async def batch_transcribe(self, audio_paths: list[str], **settings: Any) -> dict[str, Any]:
        """POST /batch_transcribe – parallel ASR for explicit file list."""
        payload: dict[str, Any] = {
            "audio_paths": audio_paths,
            "model": settings.get("model", "kb-whisper-large"),
            "backend": settings.get("backend", "faster"),
            "device": settings.get("device", "auto"),
            "language": settings.get("language", "sv"),
            "diarize": settings.get("diarize", False),
            "workers": settings.get("workers", 1),
        }
        hotwords = settings.get("hotwords")
        if hotwords:
            if isinstance(hotwords, str):
                payload["hotwords"] = [w.strip() for w in hotwords.split(",") if w.strip()]
            else:
                payload["hotwords"] = hotwords
        return await self._post("/batch_transcribe", payload)

    async def scan_process(
        self,
        directory: str,
        *,
        state_file: str | None = None,
        operation: str = "transcribe",
        batch_size: int = 4,
        max_files: int | None = None,
        **settings: Any,
    ) -> dict[str, Any]:
        """POST /scan_process – incremental directory scan with optional state file."""
        payload: dict[str, Any] = {
            "directory": directory,
            "operation": operation,
            "batch_size": batch_size,
            "model": settings.get("model", "kb-whisper-large"),
            "backend": settings.get("backend", "faster"),
            "device": settings.get("device", "auto"),
            "language": settings.get("language", "sv"),
            "diarize": settings.get("diarize", False),
            "workers": settings.get("workers", 1),
        }
        if state_file:
            payload["state_file"] = state_file
        if max_files is not None:
            payload["max_files"] = max_files
        hotwords = settings.get("hotwords")
        if hotwords:
            if isinstance(hotwords, str):
                payload["hotwords"] = [w.strip() for w in hotwords.split(",") if w.strip()]
            else:
                payload["hotwords"] = hotwords
        return await self._post("/scan_process", payload)