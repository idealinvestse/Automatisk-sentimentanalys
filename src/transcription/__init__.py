"""Transcription module for ASR backends (Faster-Whisper and HuggingFace Transformers)."""

from __future__ import annotations

from .base import Transcriber
from .factory import get_transcriber
from .faster_whisper import FasterWhisperTranscriber
from .transformers import TransformersTranscriber
from .whisperx import WhisperXTranscriber

__all__ = [
    "Transcriber",
    "get_transcriber",
    "FasterWhisperTranscriber",
    "TransformersTranscriber",
    "WhisperXTranscriber",
]
