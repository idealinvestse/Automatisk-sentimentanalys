"""Factory for obtaining Transcriber instances."""

from __future__ import annotations

from .base import Transcriber
from .faster_whisper import FasterWhisperTranscriber
from .transformers import TransformersTranscriber


def get_transcriber(
    backend: str = "faster",
    model_name: str = "kb-whisper-large",
    device: str = "auto",
) -> Transcriber:
    """Get a transcriber instance for the given backend.

    Args:
        backend: 'faster' for Faster-Whisper, or 'transformers' for HuggingFace Transformers.
        model_name: Model name or HuggingFace ID.
        device: Device specification (e.g. 'auto', 'cpu', 'cuda', 'mps').

    Returns:
        An object implementing the Transcriber protocol.

    Raises:
        ValueError: If an unknown backend is specified.
    """
    b = str(backend).strip().lower()
    if b == "faster":
        return FasterWhisperTranscriber(model_name=model_name, device=device)
    elif b == "transformers":
        return TransformersTranscriber(model_name=model_name, device=device)
    else:
        raise ValueError(
            f"Unknown transcription backend '{backend}'. Supported: 'faster', 'transformers'"
        )
