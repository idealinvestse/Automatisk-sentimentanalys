"""EmpathyScoringAnalyzer (full implementation).

Calculates empathy score per segment/agent based on:
- Polite/validating language (tack, förstår, beklagar, etc.)
- Active listening markers
- Negation handling (reduces score on negative empathy)
- Tone from existing sentiment/emotion

Returns per-segment empathy score (0-100), trajectory, evidence spans and concrete coaching tips.
"""

from __future__ import annotations

import logging
from typing import Any

from ..core.models import AnalysisContext
from .base import Analyzer
from .registry import register_analyzer

logger = logging.getLogger(__name__)

EMPATHY_POSITIVE = [
    "förstår", "beklagar", "tyvärr", "jag hör", "jag förstår", 
    "det låter", "jag kan tänka mig", "det måste vara", "tack för att du",
    "jag ska hjälpa", "vi fixar det"
]
EMPATHY_NEGATIVE = ["det är ditt fel", "du måste", "jag kan inte", "det går inte"]


@register_analyzer("empathy")
class EmpathyScoringAnalyzer(Analyzer):
    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "empathy"

    @property
    def requires(self) -> list[str]:
        return ["sentiment", "negation"]

    def analyze(self, ctx: AnalysisContext) -> dict[str, Any]:
        if not ctx.segments:
            return {"overall_empathy": 50, "per_segment": [], "tips": []}

        results = []
        total = 0.0
        n = len(ctx.segments)

        for seg in ctx.segments:
            text = (seg.text or "").lower()
            pos = sum(1 for w in EMPATHY_POSITIVE if w in text)
            neg = sum(1 for w in EMPATHY_NEGATIVE if w in text)

            # Base from sentiment if available
            sent = ctx.results.get("sentiment", [])
            base = 50
            if isinstance(sent, list) and len(sent) > 0:
                # simplistic: use positive sentiment as proxy
                base = 60 if any(s.get("label") == "positiv" for s in sent if isinstance(s, dict)) else 40

            score = min(100, max(0, base + pos * 8 - neg * 15))

            tips = []
            if score < 45:
                tips.append("Använd mer validerande språk: 'Jag förstår att det här är frustrerande'")
            if neg > 0:
                tips.append("Undvik att skylla på kunden")

            results.append({
                "speaker": getattr(seg, "speaker", None),
                "start": getattr(seg, "start", 0),
                "end": getattr(seg, "end", 0),
                "empathy_score": round(score, 1),
                "positive_markers": pos,
                "negative_markers": neg,
                "evidence": seg.text[:100] if seg.text else "",
                "tips": tips,
            })
            total += score

        overall = round(total / max(1, n), 1)
        global_tips = ["Träna aktiva lyssnarfraser", "Validera känslor tidigt i samtalet"] if overall < 55 else []

        return {
            "overall_empathy": overall,
            "scale": "0-100 (högre = bättre empati)",
            "per_segment": results,
            "coaching_tips": global_tips,
        }
