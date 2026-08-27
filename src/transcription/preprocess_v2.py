"""Deprecated alias for :mod:`src.transcription.preprocess_callcenter`."""

import warnings

from .preprocess_callcenter import preprocess_audio_callcenter

warnings.warn(
    "src.transcription.preprocess_v2 is deprecated; use src.transcription.preprocess_callcenter instead",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["preprocess_audio_callcenter"]
