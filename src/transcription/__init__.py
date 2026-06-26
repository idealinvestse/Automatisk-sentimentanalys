"""Transcription module for ASR backends (Faster-Whisper and HuggingFace Transformers)."""

from __future__ import annotations

from .base import Transcriber
from .factory import get_transcriber, resolve_preprocess_mode
from .faster_whisper import FasterWhisperTranscriber
from .transformers import TransformersTranscriber
from .whisperx import WhisperXTranscriber

__all__ = [
    "Transcriber",
    "get_transcriber",
    "resolve_preprocess_mode",
    "FasterWhisperTranscriber",
    "TransformersTranscriber",
    "WhisperXTranscriber",
]
