"""Spoken text normalizer (Task 2.5) – post ASR cleaning for analysis.

Removes fillers, normalizes repetitions etc. Does not mutate the original strict transcript.
When selected, runs before sentiment/intent; downstream adapters read normalized text via
``segment_analysis_text`` in ``text_utils``.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ..core.models import AnalysisContext, Segment
from .base import Analyzer
from .registry import register_analyzer

logger = logging.getLogger(__name__)

FILLERS = re.compile(r"\b(eh|hmm|öh|du vet|typ|liksom|asså|alltså)\b", re.IGNORECASE)
_ORPHAN_COMMA_RUN = re.compile(r"(?:,\s*)+")


def normalize_spoken_text(text: str) -> str:
    """Strip Swedish ASR fillers and collapse punctuation left behind."""
    clean = FILLERS.sub(" ", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    clean = _ORPHAN_COMMA_RUN.sub(", ", clean)
    clean = re.sub(r"^,\s*", "", clean)
    clean = re.sub(r"\s*,\s*$", "", clean)
    return re.sub(r"\s+", " ", clean).strip()


@register_analyzer("spoken_normalizer")
class SpokenNormalizerAnalyzer(Analyzer):
    @property
    def name(self) -> str:
        return "spoken_normalizer"

    @property
    def requires(self) -> list[str]:
        return []

    def analyze(self, ctx: AnalysisContext) -> list[dict[str, Any]]:
        normalized = []
        for seg in ctx.segments or []:
            text = seg.text or ""
            clean = normalize_spoken_text(text)
            normalized.append({"original": text, "normalized": clean, "speaker": seg.speaker})
        return normalized
