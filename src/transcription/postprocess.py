"""Post-processing filters for ASR transcripts (hallucination removal)."""

from __future__ import annotations

import re
from dataclasses import replace

from ..core.models import Segment, Transcript

_HALLUCINATION_PATTERNS: tuple[str, ...] = (
    "thanks for watching",
    "thank you for watching",
    "subscribe to",
    "tack för att ni tittade",
    "textning av",
    "undertexter av",
)

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.strip().lower())


def _is_repetition_loop(text: str) -> bool:
    tokens = text.split()
    return len(tokens) >= 4 and len(set(tokens)) == 1


def _is_hallucination(text: str) -> bool:
    normalized = _normalize(text)
    if not normalized:
        return True
    if any(pattern in normalized for pattern in _HALLUCINATION_PATTERNS):
        return True
    return _is_repetition_loop(normalized)


def filter_hallucinations(transcript: Transcript) -> Transcript:
    """Return a new transcript with Whisper ghost segments removed."""
    kept: list[Segment] = []
    dropped = 0

    for segment in transcript.segments:
        if _is_hallucination(segment.text):
            dropped += 1
        else:
            kept.append(segment)

    warnings = list(transcript.warnings)
    if dropped > 0:
        warnings.append(f"hallucination_dropped:{dropped}")

    return replace(transcript, segments=kept, warnings=warnings)
