"""UpsellOpportunityDetector.

Detects moments in the conversation where upsell or cross-sell is natural and low-friction:
- Positive sentiment + expressed need
- Specific product/service mentions
- Buying signals ("jag skulle kunna", "kanske bättre att", "har ni något för")

Returns flagged segments with confidence and suggested next action.
High business value for revenue teams.
"""

from __future__ import annotations

import logging
from typing import Any

from ..core.models import AnalysisContext
from .base import Analyzer
from .registry import register_analyzer

logger = logging.getLogger(__name__)

BUYING_SIGNALS = [
    "jag skulle kunna", "kanske bättre", "har ni något", "vad kostar", 
    "kan man uppgradera", "finns det något bättre", "jag behöver också"
]
POSITIVE_CONTEXT = ["bra", "funkar", "nöjd", "perfekt", "tack"]


@register_analyzer("upsell_opportunity")
class UpsellOpportunityDetector(Analyzer):
    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "upsell_opportunity"

    @property
    def requires(self) -> list[str]:
        return ["sentiment"]

    def analyze(self, ctx: AnalysisContext) -> dict[str, Any]:
        if not ctx.segments:
            return {"opportunities": [], "count": 0}

        opportunities = []

        for seg in ctx.segments:
            text = (seg.text or "").lower()
            score = 0
            signals = []

            if any(sig in text for sig in BUYING_SIGNALS):
                score += 40
                signals.append("buying_signal")
            if any(pos in text for pos in POSITIVE_CONTEXT):
                score += 25
                signals.append("positive_context")

            # Boost if previous sentiment was positive
            sent = ctx.results.get("sentiment", [])
            if isinstance(sent, list) and sent:
                last = sent[-1] if isinstance(sent[-1], dict) else {}
                if last.get("label") == "positiv":
                    score += 20

            if score >= 50:
                opportunities.append({
                    "speaker": getattr(seg, "speaker", None),
                    "start": getattr(seg, "start", 0),
                    "end": getattr(seg, "end", 0),
                    "confidence": min(95, score),
                    "signals": signals,
                    "suggested_action": "Erbjud relevant uppgradering eller tilläggstjänst här",
                    "evidence": seg.text[:90] if seg.text else "",
                })

        return {
            "opportunities": opportunities,
            "count": len(opportunities),
            "recommendation": "Träna agenter att agera på dessa tillfällen" if opportunities else "Inga tydliga tillfällen upptäckta",
        }
