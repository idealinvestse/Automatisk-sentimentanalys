"""Spoken text normalizer (Task 2.5) – post ASR cleaning for analysis.

Removes fillers, normalizes repetitions etc. Does not mutate the original strict transcript.
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
            clean = FILLERS.sub("", text)
            clean = re.sub(r"\s+", " ", clean).strip()
            normalized.append({"original": text, "normalized": clean, "speaker": seg.speaker})
        return normalized
