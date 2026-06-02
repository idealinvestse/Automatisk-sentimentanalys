"""Factory for obtaining cached Transcriber instances."""

from __future__ import annotations

import logging
from functools import lru_cache

from .base import Transcriber
from .faster_whisper import FasterWhisperTranscriber
from .transformers import TransformersTranscriber
# whisperx is imported lazily inside the factory so that the package is only
# required when the user actually selects backend="whisperx".

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _get_cached_transcriber(backend: str, model_name: str, device: str) -> Transcriber:
    """Return a cached Transcriber instance for the given (backend, model, device) triple.

    The ``lru_cache`` ensures that loading a large ASR model (which can take
    10–30 seconds) only happens once per unique configuration per process
    lifetime.  A cache of 4 slots covers the most common patterns (e.g.
    kb-whisper-large on CPU + one alternate model).

    Args:
        backend: ``'faster'`` for Faster-Whisper, ``'transformers'`` for HF Transformers,
                 ``'whisperx'`` for WhisperX (alignment + optional diarization).
        model_name: Model name or HuggingFace ID.
        device: Device specification (e.g. ``'auto'``, ``'cpu'``, ``'cuda'``).

    Returns:
        A cached object implementing the :class:`Transcriber` protocol.

    Raises:
        ValueError: If an unknown backend is specified.
    """
    b = str(backend).strip().lower()
    logger.info("Loading ASR transcriber: backend=%s model=%s device=%s", b, model_name, device)
    if b == "faster":
        return FasterWhisperTranscriber(model_name=model_name, device=device)
    if b == "transformers":
        return TransformersTranscriber(model_name=model_name, device=device)
    if b == "whisperx":
        # Lazy import keeps whisperx an optional dependency
        from .whisperx import WhisperXTranscriber

        return WhisperXTranscriber(model_name=model_name, device=device)
    raise ValueError(
        f"Unknown transcription backend '{backend}'. Supported: 'faster', 'transformers', 'whisperx'"
    )


def clear_transcriber_cache() -> None:
    """Clear the internal transcriber cache.

    Call this between tests or when you need to force a fresh model load
    (e.g. after changing environment variables that affect model selection).
    """
    _get_cached_transcriber.cache_clear()


def get_transcriber(
    backend: str = "faster",
    model_name: str = "kb-whisper-large",
    device: str = "auto",
) -> Transcriber:
    """Get a (cached) transcriber instance for the given backend.

    Delegates to :func:`_get_cached_transcriber` so that the same
    (backend, model_name, device) triple always returns the same object
    without reloading the model weights.

    Args:
        backend: ``'faster'`` for Faster-Whisper (recommended default for Swedish WER),
                 ``'transformers'`` for HF Transformers pipeline,
                 ``'whisperx'`` for WhisperX (superior word alignment + integrated diarization).
        model_name: Model name or HuggingFace ID.
        device: Device specification (e.g. ``'auto'``, ``'cpu'``, ``'cuda'``, ``'mps'``).

    Returns:
        An object implementing the :class:`Transcriber` protocol.

    Raises:
        ValueError: If an unknown backend is specified.
    """
    return _get_cached_transcriber(
        backend=str(backend).strip().lower(),
        model_name=model_name,
        device=device,
    )
