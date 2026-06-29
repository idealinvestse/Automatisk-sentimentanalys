"""Customer Effort Score (CES) Analyzer.

Detects friction points that increase customer effort:
- Filler words & hesitations ("eh", "alltså", "typ")
- Repetitions and requests for clarification
- Long or complex turns

Returns per-segment effort indicators + overall score + coaching tips.
Useful for process improvement and agent clarity training.
"""

from __future__ import annotations

import logging
from typing import Any

from ..core.models import AnalysisContext
from .base import Analyzer
from .registry import register_analyzer

logger = logging.getLogger(__name__)

FILLER_WORDS = ["eh", "öh", "alltså", "typ", "liksom", "va", "så att säga", "du vet"]
CLARIFICATION = ["vad menar du", "hur då", "kan du upprepa", "jag förstår inte", "va?"]


@register_analyzer("customer_effort")
class CustomerEffortScoreAnalyzer(Analyzer):
    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "customer_effort"

    @property
    def requires(self) -> list[str]:
        return []

    def analyze(self, ctx: AnalysisContext) -> dict[str, Any]:
        if not ctx.segments:
            return {
                "overall_ces": 0,
                "scale": "0-100 (högre = mer effort/frustration)",
                "per_segment": [],
                "coaching_tips": [],
            }

        results = []
        total_effort = 0.0
        n = len(ctx.segments)

        for seg in ctx.segments:
            text = (seg.text or "").lower()
            fillers = sum(text.count(w) for w in FILLER_WORDS)
            clarifs = sum(1 for m in CLARIFICATION if m in text)
            # Naive repetition
            words = [w for w in text.split() if len(w) > 2]
            reps = max(0, len(words) - len(set(words)))
            dur = getattr(seg, "end", 0) - getattr(seg, "start", 0)
            effort = min(100.0, fillers * 7 + clarifs * 12 + reps * 2.5 + max(0, dur - 20) * 0.6)
            total_effort += effort

            results.append(
                {
                    "speaker": getattr(seg, "speaker", None),
                    "start": getattr(seg, "start", 0),
                    "end": getattr(seg, "end", 0),
                    "effort_score": round(effort, 1),
                    "fillers": fillers,
                    "clarifications": clarifs,
                    "repetitions": reps,
                    "duration_s": round(dur, 1),
                }
            )

        overall = round(total_effort / max(1, n), 1)
        tips = []
        if overall > 40:
            tips = ["Minska filler words", "Bekräfta förståelse tidigt", "Håll svar kortare"]

        return {
            "overall_ces": overall,
            "scale": "0-100 (högre = mer effort/frustration)",
            "per_segment": results,
            "coaching_tips": tips,
        }
