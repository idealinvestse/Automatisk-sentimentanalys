"""Granulär multi-label emotion analysis (Task 2.1).

Emotions: frustration, ilska, besvikelse, förvirring, tillfredsställelse, neutral, oro, glädje.

Hybrid: keyword markers (svenska) + per-segment sentiment polarity + negation dampening.
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
    "ilska": ["rasande", "ilska", "förbannad", "helvete"],
    "besvikelse": ["besviken", "trist", "ledsen", "synd", "inte bra"],
    "förvirring": ["förvirrad", "förstår inte", "vet inte", "oklart", "konstigt"],
    "tillfredsställelse": ["perfekt", "nöjd", "funkar"],
    "oro": ["orolig", "rädd", "osäker", "vad händer", "problem"],
    "glädje": ["glad", "kul", "super"],
}

# Shared positive words — lower weight to avoid overlap noise
_POSITIVE_SHARED = frozenset({"bra", "tack"})

EMOTION_REGEX = {
    e: re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b", re.IGNORECASE)
    for e, keys in EMOTION_KEYWORDS.items()
}

_NEGATIVE_SENTIMENT = frozenset({"negativ", "negative"})
_POSITIVE_SENTIMENT = frozenset({"positiv", "positive"})


@register_analyzer("emotion")
class EmotionAnalyzer(Analyzer):
    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "emotion"

    @property
    def requires(self) -> list[str]:
        return ["sentiment", "negation"]

    def analyze(self, ctx: AnalysisContext) -> list[dict[str, Any]]:
        sentiment_results = ctx.results.get("sentiment") or []
        negation_results = ctx.results.get("negation") or []
        out = []
        for idx, seg in enumerate(ctx.segments or []):
            text = (seg.text or "").lower()
            scores: dict[str, float] = {}
            for emotion, regex in EMOTION_REGEX.items():
                if regex.search(text):
                    scores[emotion] = 0.75

            sent = sentiment_results[idx] if idx < len(sentiment_results) else {}
            label = str(sent.get("label", "neutral")).lower() if isinstance(sent, dict) else "neutral"
            if label in _NEGATIVE_SENTIMENT:
                for emo in ("frustration", "besvikelse", "oro"):
                    scores[emo] = max(scores.get(emo, 0.0), 0.55)
            elif label in _POSITIVE_SENTIMENT:
                for emo in ("tillfredsställelse", "glädje"):
                    scores[emo] = max(scores.get(emo, 0.0), 0.55)

            neg = negation_results[idx] if idx < len(negation_results) else {}
            if isinstance(neg, dict) and neg.get("has_negation"):
                for emo in list(scores):
                    if emo in ("tillfredsställelse", "glädje"):
                        scores[emo] *= 0.5

            if not scores:
                scores = {"neutral": 0.9}
            primary = max(scores, key=scores.get)
            out.append({
                "primary": primary,
                "scores": scores,
                "speaker": getattr(seg, "speaker", None),
            })
        return out
