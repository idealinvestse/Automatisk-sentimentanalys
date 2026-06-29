"""httpx async client for the FastAPI backend.

Fas 3 – docs/archive/MIGRATION_TO_NICEGUI_PLAN.md §3
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

_ASR_KEYS = (
    "model",
    "backend",
    "device",
    "language",
    "beam_size",
    "vad",
    "chunk_length_s",
    "revision",
    "diarize",
    "num_speakers",
    "initial_prompt",
    "word_timestamps",
    "preprocess",
    "workers",
    "worker_timeout",
    "sentiment_profile",
)


def _asr_payload(settings: dict[str, Any]) -> dict[str, Any]:
    """Map dashboard settings to API AsrParamsMixin + batch fields."""
    payload: dict[str, Any] = {}
    for key in _ASR_KEYS:
        if key in settings and settings[key] is not None:
            payload[key] = settings[key]
    hotwords = settings.get("hotwords")
    if hotwords:
        if isinstance(hotwords, str):
            payload["hotwords"] = [w.strip() for w in hotwords.split(",") if w.strip()]
        else:
            payload["hotwords"] = hotwords
    return payload


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

    async def _get(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=self._headers(), params=params)
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

    async def get_process_events(self, *, limit: int = 100) -> dict[str, Any]:
        """Fetch recent process status events from ``GET /status/processes``."""
        return await self._get("/status/processes", params={"limit": limit})

    async def _delete(self, path: str) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.delete(url, headers=self._headers())
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
        profile: str = "callcenter",
        selected_analyzers: list[str] | None = None,
        async_analyzers: bool = False,
        use_mistral_llm: bool = False,
        deep_analysis: bool = False,
        device: str = "auto",
        llm_model: str | None = None,
        provider: str = "openrouter",
        groq_eu_residency: bool = False,
    ) -> dict[str, Any]:
        """POST /analyze_pipeline – full CallAnalysisPipeline on segments."""
        payload: dict[str, Any] = {
            "segments": segments,
            "profile": profile,
            "async_analyzers": async_analyzers,
            "device": device,
            "use_mistral_llm": use_mistral_llm,
            "deep_analysis": deep_analysis,
            "provider": provider,
            "groq_eu_residency": groq_eu_residency,
        }
        if selected_analyzers:
            payload["selected_analyzers"] = selected_analyzers
        if llm_model:
            payload["llm_model"] = llm_model
        return await self._post("/analyze_pipeline", payload)

    async def get_alerting_status(self) -> dict[str, Any]:
        """GET /alerting/status – webhook and circuit breaker health."""
        return await self._get("/alerting/status")

    async def analyze_text(
        self,
        texts: list[str],
        *,
        profile: str | None = None,
        device: str = "auto",
        return_all_scores: bool = True,
    ) -> dict[str, Any]:
        """POST /analyze – text sentiment on one or more strings."""
        payload: dict[str, Any] = {
            "texts": texts,
            "device": device,
            "return_all_scores": return_all_scores,
        }
        if profile:
            payload["profile"] = profile
        return await self._post("/analyze", payload)

    async def analyze_conversation(
        self,
        audio_path: str,
        *,
        use_full_pipeline: bool = False,
        model: str = "kb-whisper-large",
        backend: str = "faster",
        device: str = "auto",
        language: str = "sv",
        diarize: bool = False,
        sentiment_profile: str = "callcenter",
    ) -> dict[str, Any]:
        """POST /analyze_conversation – transcribe (+ optional full pipeline) on audio."""
        payload: dict[str, Any] = {
            "audio_path": audio_path,
            "model": model,
            "backend": backend,
            "device": device,
            "language": language,
            "diarize": diarize,
            "use_full_pipeline": use_full_pipeline,
            "sentiment_profile": sentiment_profile,
            "return_all_scores": True,
        }
        return await self._post("/analyze_conversation", payload)

    async def transcribe(self, audio_path: str, **settings: Any) -> dict[str, Any]:
        """POST /transcribe – single-file ASR."""
        payload = {"audio_path": audio_path, **_asr_payload(settings)}
        if "preprocess" not in payload:
            payload["preprocess"] = settings.get("preprocess", True)
        return await self._post("/transcribe", payload)

    async def batch_transcribe(self, audio_paths: list[str], **settings: Any) -> dict[str, Any]:
        """POST /batch_transcribe – parallel ASR for explicit file list."""
        payload = {"audio_paths": audio_paths, **_asr_payload(settings)}
        if "workers" not in payload:
            payload["workers"] = settings.get("workers", 1)
        return await self._post("/batch_transcribe", payload)

    async def scan_process(
        self,
        directory: str,
        *,
        state_file: str | None = None,
        operation: str = "transcribe",
        batch_size: int = 4,
        max_files: int | None = None,
        pattern: str | None = None,
        recursive: bool = True,
        **settings: Any,
    ) -> dict[str, Any]:
        """POST /scan_process – incremental directory scan with optional state file."""
        payload: dict[str, Any] = {
            "directory": directory,
            "operation": operation,
            "batch_size": batch_size,
            "recursive": recursive,
            **_asr_payload(settings),
        }
        if state_file:
            payload["state_file"] = state_file
        if max_files is not None:
            payload["max_files"] = max_files
        if pattern:
            payload["pattern"] = pattern
        if "workers" not in payload:
            payload["workers"] = settings.get("workers", 1)
        return await self._post("/scan_process", payload)

    async def get_transcription_job(self, job_id: str) -> dict[str, Any]:
        """GET /transcription/jobs/{job_id} – job status snapshot."""
        return await self._get(f"/transcription/jobs/{job_id}")

    async def list_transcription_jobs(self, *, limit: int = 20) -> dict[str, Any]:
        """GET /transcription/jobs – recent transcription jobs."""
        return await self._get("/transcription/jobs", params={"limit": limit})

    async def cancel_job(self, job_id: str) -> dict[str, Any]:
        """POST /transcription/jobs/{job_id}/cancel – request job cancellation."""
        return await self._post(f"/transcription/jobs/{job_id}/cancel", {})

    # ------------------------------------------------------------------
    # Fas 4 call center endpoints (agent perf, search, insights, qa, alerts)
    # ------------------------------------------------------------------

    async def get_agent_performance(
        self,
        agent_id: str,
        segments_list: list[list[dict[str, Any]]],
        *,
        window: str = "7d",
        profile: str = "callcenter",
        reanalyze: bool = False,
        use_mistral_llm: bool = False,
        deep_analysis: bool = False,
        llm_model: str | None = None,
    ) -> dict[str, Any]:
        """POST /agent_performance/{agent_id} – cached agent aggregates."""
        payload: dict[str, Any] = {
            "agent_id": agent_id,
            "segments_list": segments_list,
            "window": window,
            "profile": profile,
            "reanalyze": reanalyze,
            "use_mistral_llm": use_mistral_llm,
            "deep_analysis": deep_analysis,
        }
        if llm_model:
            payload["llm_model"] = llm_model
        return await self._post(f"/agent_performance/{agent_id}", payload)

    async def semantic_search(
        self,
        query: str,
        segments_list: list[list[dict[str, Any]]],
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        profile: str = "callcenter",
        reanalyze: bool = False,
    ) -> dict[str, Any]:
        """POST /search/semantic – hybrid search over calls."""
        payload: dict[str, Any] = {
            "query": query,
            "segments_list": segments_list,
            "top_k": top_k,
            "profile": profile,
            "reanalyze": reanalyze,
        }
        if filters:
            payload["filters"] = filters
        return await self._post("/search/semantic", payload)

    async def get_hot_topics(
        self,
        segments_list: list[list[dict[str, Any]]],
        *,
        window: str = "7d",
        profile: str = "callcenter",
        reanalyze: bool = False,
    ) -> dict[str, Any]:
        """POST /insights/hot_topics – aggregated hot topics."""
        return await self._post(
            "/insights/hot_topics",
            {
                "segments_list": segments_list,
                "window": window,
                "profile": profile,
                "reanalyze": reanalyze,
            },
        )

    async def score_qa(
        self,
        segments: list[dict[str, Any]],
        *,
        profile: str = "callcenter",
        use_mistral_llm: bool = False,
        deep_analysis: bool = False,
        provider: str = "openrouter",
    ) -> dict[str, Any]:
        """POST /qa/score – compliance QA scorecard for one call."""
        return await self._post(
            "/qa/score",
            {
                "segments": segments,
                "profile": profile,
                "use_mistral_llm": use_mistral_llm,
                "deep_analysis": deep_analysis,
                "provider": provider,
            },
        )

    async def get_alerts(
        self,
        *,
        segments_list: list[list[dict[str, Any]]] | None = None,
        aggregate: dict[str, Any] | None = None,
        profile: str = "callcenter",
        reanalyze: bool = False,
    ) -> dict[str, Any]:
        """POST /alerts – per-call or aggregate trend alerts."""
        payload: dict[str, Any] = {"profile": profile, "reanalyze": reanalyze}
        if segments_list is not None:
            payload["segments_list"] = segments_list
        if aggregate is not None:
            payload["aggregate"] = aggregate
        return await self._post("/alerts", payload)