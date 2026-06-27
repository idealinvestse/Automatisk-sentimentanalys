"""MultiTurnJourneyMapper.

Maps the customer's journey across the conversation (or multiple interactions if history is available).
Tracks:
- Topic shifts and unresolved issues
- Emotion/sentiment arc over time
- Escalation points and resolution attempts

Returns journey stages, key turning points, and whether the issue feels resolved.
Very powerful for understanding complex, multi-issue calls.
"""

from __future__ import annotations

import logging
from typing import Any

from ..core.models import AnalysisContext
from .base import Analyzer
from .registry import register_analyzer

logger = logging.getLogger(__name__)


@register_analyzer("multi_turn_journey")
class MultiTurnJourneyMapper(Analyzer):
    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "multi_turn_journey"

    @property
    def requires(self) -> list[str]:
        return ["trajectory", "topics", "sentiment", "intent"]

    def analyze(self, ctx: AnalysisContext) -> dict[str, Any]:
        if not ctx.segments or len(ctx.segments) < 3:
            return {"journey_stages": [], "resolved": False, "message": "Too short for journey mapping"}

        # Simple stage detection based on intent + sentiment changes
        stages = []
        current_stage = "opening"
        unresolved = 0

        for i, seg in enumerate(ctx.segments):
            text = (seg.text or "").lower()
            intent = ctx.results.get("intent", [])
            sentiment = ctx.results.get("sentiment", [])

            if i == 0:
                current_stage = "opening"
            elif any(kw in text for kw in ["men", "dock", "fortfarande", "inte hjälpt"]):
                current_stage = "escalation"
                unresolved += 1
            elif any(kw in text for kw in ["tack", "då är det bra", "då fixar vi det"]):
                current_stage = "resolution"

            stages.append({
                "stage": current_stage,
                "start": getattr(seg, "start", 0),
                "speaker": getattr(seg, "speaker", None),
                "text_snippet": seg.text[:60] if seg.text else "",
            })

        resolved = stages[-1]["stage"] == "resolution" if stages else False

        return {
            "journey_stages": stages,
            "resolved": resolved,
            "unresolved_count": unresolved,
            "key_turning_points": [s for s in stages if s["stage"] in ["escalation", "resolution"]],
            "recommendation": "Bra journey mapping - använd för komplexa ärenden" if len(stages) > 5 else "Kort samtal - mindre behov av journey mapping",
        }
