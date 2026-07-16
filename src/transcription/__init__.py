"""Transcription module for ASR backends (Faster-Whisper and HuggingFace Transformers)."""

from __future__ import annotations

from .base import Transcriber
from .factory import get_transcriber, resolve_preprocess_mode
from .faster_whisper import FasterWhisperTranscriber
from .router import AsrRouter, get_asr_router, resolve_asr_provider, transcribe_with_router
from .transformers import TransformersTranscriber
from .whisperx import WhisperXTranscriber

__all__ = [
    "AsrRouter",
    "Transcriber",
    "get_asr_router",
    "get_transcriber",
    "resolve_asr_provider",
    "resolve_preprocess_mode",
    "transcribe_with_router",
    "FasterWhisperTranscriber",
    "TransformersTranscriber",
    "WhisperXTranscriber",
]
