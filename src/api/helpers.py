"""Shared helper functions for API router handlers."""

from __future__ import annotations

import logging
from typing import Any

from ..transcription import get_transcriber
from ..transcription.factory import resolve_preprocess_mode

logger = logging.getLogger(__name__)


def asr_kwargs_from(
    req: object,
    *,
    audio_path: str | None = None,
    word_timestamps: bool | None = None,
    preprocess: bool = False,
) -> dict[str, Any]:
    """Build keyword arguments for :func:`transcribe_helper` from an ASR request model."""
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
        "hotwords": getattr(req, "hotwords", None),
        "initial_prompt": getattr(req, "initial_prompt", None),
        "preprocess": preprocess,
        "preprocess_mode": getattr(req, "preprocess_mode", None),
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
) -> dict[str, Any]:
    """Run ASR transcription and return the result as a plain dict.

    Uses the cached :func:`~src.transcription.get_transcriber` so that model
    weights are only loaded once per unique (backend, model, device) triple.

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

    Returns:
        Transcription result as a plain dict (via ``Transcript.to_dict()``).
    """
    transcriber = get_transcriber(backend=backend, model_name=model, device=device)
    resolved_preprocess_mode = resolve_preprocess_mode(
        preprocess=preprocess,
        preprocess_mode=preprocess_mode,
        profile=profile,
    )
    transcript = transcriber.transcribe(
        audio_path=audio_path,
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
    return transcript.to_dict()
