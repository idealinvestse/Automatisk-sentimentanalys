"""Shared helper functions for API router handlers."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException

from ..core.errors import (
    ASR_DECODE_FAILED,
    ASR_EMPTY_TRANSCRIPT,
    ASR_INIT_FAILED,
    TranscriptionError,
)
from ..customers import (
    CustomerContext,
    CustomerExecutionPolicy,
    CustomerPolicyError,
    CustomerResolutionError,
    clamp_execution_policy,
    get_customer_registry,
    resolve_customer,
)
from ..pipeline import CallAnalysisPipeline
from ..transcription.factory import resolve_preprocess_mode
from ..transcription.router import AsrRouter
from .schemas import AsrParamsMixin

logger = logging.getLogger(__name__)


def resolve_customer_context(filename: str | None) -> CustomerContext | None:
    """Resolve the customer routing context for an audio filename.

    Raises :class:`CustomerResolutionError` subclasses in controlled modes
    (batch workers catch these per file). Returns ``None`` when the registry
    is disabled, or when mode is ``optional`` and the name carries no ID.
    """
    return resolve_customer(filename, get_customer_registry())


def resolve_customer_or_422(filename: str | None) -> CustomerContext | None:
    """Resolve customer context, mapping resolution errors to HTTP 422."""
    try:
        return resolve_customer_context(filename)
    except CustomerResolutionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def apply_customer_policy(
    customer: CustomerContext | None,
    *,
    requested_asr_provider: str = "local",
    requested_cloud_fallback: bool = False,
    requested_llm_enabled: bool = False,
    requested_llm_provider: str | None = None,
    requested_profile: str | None = None,
) -> CustomerExecutionPolicy:
    """Clamp request parameters to the customer ceiling; widening is HTTP 422."""
    try:
        return clamp_execution_policy(
            customer,
            requested_asr_provider=requested_asr_provider,
            requested_cloud_fallback=requested_cloud_fallback,
            requested_llm_enabled=requested_llm_enabled,
            requested_llm_provider=requested_llm_provider,
            requested_profile=requested_profile,
        )
    except CustomerPolicyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def apply_llm_ceiling(pipe: CallAnalysisPipeline, policy: CustomerExecutionPolicy) -> None:
    """Keep profile defaults from re-enabling LLM beyond the customer ceiling."""
    if not policy.llm_enabled:
        pipe.use_mistral_llm = False
        pipe.deep_analysis = False
        return
    if policy.llm_provider:
        pipe.provider = policy.llm_provider
        pipe.use_mistral_llm = True


def usable_segment_count(transcript: Any) -> int:
    """Count segments that contain non-empty speech text."""
    if transcript is None:
        return 0
    if isinstance(transcript, dict):
        segments = transcript.get("segments") or []
    else:
        segments = getattr(transcript, "segments", None) or []
    count = 0
    for segment in segments:
        text = segment.get("text") if isinstance(segment, dict) else getattr(segment, "text", "")
        if str(text or "").strip():
            count += 1
    return count


def require_usable_transcript(transcript: Any) -> None:
    """Fail closed when ASR produced no usable speech."""
    if usable_segment_count(transcript) == 0:
        raise TranscriptionError(
            "Transcription contained no speech that can be analyzed.",
            error_code=ASR_EMPTY_TRANSCRIPT,
        )


def classify_asr_error(exc: BaseException) -> str:
    """Map an ASR exception to a stable API error code."""
    message = str(exc).lower()
    if isinstance(exc, ImportError):
        return ASR_INIT_FAILED
    if any(token in message for token in ("cuda", "cublas", "out of memory", "oom")):
        return ASR_INIT_FAILED
    if any(token in message for token in ("failed to load", "download", "hub", "checkpoint")):
        return ASR_INIT_FAILED
    return ASR_DECODE_FAILED


def asr_kwargs_from(
    req: AsrParamsMixin,
    *,
    audio_path: str | None = None,
    word_timestamps: bool | None = None,
    preprocess: bool = False,
    customer: CustomerContext | None = None,
    requested_profile: str | None = None,
) -> dict[str, Any]:
    """Build keyword arguments for :func:`transcribe_helper` from an ASR request model."""
    policy = apply_customer_policy(
        customer,
        requested_asr_provider=req.provider,
        requested_cloud_fallback=req.cloud_fallback_local,
        requested_profile=requested_profile,
    )
    kwargs: dict[str, Any] = {
        "model": req.model,
        "backend": req.backend,
        "device": req.device,
        "language": req.language,
        "beam_size": req.beam_size,
        "vad": req.vad,
        "chunk_length_s": req.chunk_length_s,
        "revision": req.revision,
        "diarize": req.diarize,
        "num_speakers": req.num_speakers,
        "provider": policy.asr_provider,
        "cloud_fallback_local": policy.cloud_fallback_local,
        "hotwords": getattr(req, "hotwords", None),
        "initial_prompt": getattr(req, "initial_prompt", None),
        "preprocess": preprocess,
        "preprocess_mode": getattr(req, "preprocess_mode", None),
        "profile": policy.analyzer_profile,
    }
    if audio_path is not None:
        kwargs["audio_path"] = audio_path
    wt = word_timestamps if word_timestamps is not None else getattr(req, "word_timestamps", True)
    kwargs["word_timestamps"] = wt
    return kwargs


def transcribe_helper(
    audio_path: str,
    model: str = "kb-whisper-large",
    backend: str = "faster",
    device: str = "auto",
    language: str = "sv",
    beam_size: int = 5,
    vad: bool = True,
    word_timestamps: bool = True,
    chunk_length_s: int = 30,
    revision: str | None = None,
    diarize: bool = False,
    num_speakers: int | None = None,
    hotwords: list[str] | None = None,
    initial_prompt: str | None = None,
    preprocess: bool = False,
    preprocess_mode: str | None = None,
    profile: str | None = None,
    provider: str = "local",
    cloud_fallback_local: bool = False,
    require_speech: bool = True,
) -> dict[str, Any]:
    """Run ASR transcription and return the result as a plain dict.

    Routes through :class:`~src.transcription.router.AsrRouter` so provider
    policy and post-processing apply consistently across entry points.

    Args:
        audio_path: Path to the audio file.
        model: ASR model name or alias.
        backend: ``'faster'`` (default), ``'transformers'`` or ``'whisperx'``.
        device: Device string (``'auto'``, ``'cpu'``, ``'cuda'``, …).
        language: BCP-47 language code (default ``'sv'``).
        beam_size: Beam width for decoding.
        vad: Whether to apply Voice Activity Detection.
        word_timestamps: Whether to include word-level timestamps.
        chunk_length_s: Audio chunk length in seconds.
        revision: KB-Whisper revision (``'standard'``, ``'strict'``, ``'subtitle'``).
        diarize: Whether to run speaker diarization.
        num_speakers: Expected number of speakers.
        hotwords: Domain words to boost during ASR.
        initial_prompt: Conditioning prompt for the decoder.
        require_speech: If True, raise when the transcript has no usable speech.

    Returns:
        Transcription result as a plain dict (via ``Transcript.to_dict()``).
    """
    resolved_preprocess_mode = resolve_preprocess_mode(
        preprocess=preprocess,
        preprocess_mode=preprocess_mode,
        profile=profile,
    )
    try:
        transcript = AsrRouter().transcribe(
            audio_path,
            provider=provider,
            backend=backend,
            model_name=model,
            device=device,
            cloud_fallback_local=cloud_fallback_local,
            language=language,
            beam_size=beam_size,
            vad=vad,
            word_timestamps=word_timestamps,
            chunk_length_s=chunk_length_s,
            revision=revision,
            diarize=diarize,
            num_speakers=num_speakers,
            hotwords=hotwords,
            initial_prompt=initial_prompt,
            preprocess=preprocess,
            preprocess_mode=resolved_preprocess_mode,
        )
    except TranscriptionError:
        raise
    except Exception as exc:
        raise TranscriptionError(str(exc), error_code=classify_asr_error(exc)) from exc
    payload = transcript.to_dict()
    if require_speech:
        require_usable_transcript(payload)
    return payload
