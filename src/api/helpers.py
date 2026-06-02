"""Shared helper functions for API router handlers."""

from __future__ import annotations

import logging
from typing import Any

from ..transcription import get_transcriber

logger = logging.getLogger(__name__)


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
    )
    return transcript.to_dict()
