"""Granulär multi-label emotion analysis (Task 2.1).

Emotions: frustration, ilska, besvikelse, förvirring, tillfredsställelse, neutral, oro, glädje.

Hybrid: keyword markers (svenska) + reuse of sentiment for polarity signal.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ..core.models import AnalysisContext
from .base import Analyzer
from .registry import register_analyzer

logger = logging.getLogger(__name__)

EMOTION_KEYWORDS = {
    "frustration": ["jättearg", "trött", "irriterad", "frustrerad", "arg", "jävla", "fan"],
    "ilska": ["arg", "rasande", "ilska", "förbannad", "jävla", "helvete"],
    "besvikelse": ["besviken", "trist", "ledsen", "synd", "inte bra"],
    "förvirring": ["förvirrad", "förstår inte", "hur", "vad", "oklart", "konstigt"],
    "tillfredsställelse": ["bra", "tack", "perfekt", "nöjd", "funkar"],
    "oro": ["orolig", "rädd", "osäker", "vad händer", "problem"],
    "glädje": ["glad", "bra", "kul", "tack", "super"],
}

EMOTION_REGEX = {
    e: re.compile(r"\b(" + "|".join(k for k in keys) + r")\b", re.IGNORECASE)
    for e, keys in EMOTION_KEYWORDS.items()
}


@register_analyzer("emotion")
class EmotionAnalyzer(Analyzer):
    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "emotion"

    @property
    def requires(self) -> list[str]:
        return []

    def analyze(self, ctx: AnalysisContext) -> list[dict[str, Any]]:
        out = []
        for seg in ctx.segments or []:
            text = (seg.text or "").lower()
            scores = {}
            for emotion, regex in EMOTION_REGEX.items():
                if regex.search(text):
                    scores[emotion] = 0.75  # heuristic score
            if not scores:
                scores = {"neutral": 0.9}
            primary = max(scores, key=scores.get)
            out.append({
                "primary": primary,
                "scores": scores,
                "speaker": getattr(seg, "speaker", None),
            })
        return out
