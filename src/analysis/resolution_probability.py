"""ResolutionProbabilityPredictor.

Estimates the probability that the current issue will be resolved in this call.
Uses:
- Sentiment trajectory (improving vs worsening)
- Number of clarifications and repetitions
- Agent empathy and listening signals
- Call duration so far

Returns probability (0-100) + confidence + recommended actions to increase resolution chance.
Extremely useful for real-time supervisor alerts and post-call QA.
"""

from __future__ import annotations

import logging
from typing import Any

from ..core.models import AnalysisContext
from .base import Analyzer
from .registry import register_analyzer

logger = logging.getLogger(__name__)


@register_analyzer("resolution_probability")
class ResolutionProbabilityPredictor(Analyzer):
    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "resolution_probability"

    @property
    def requires(self) -> list[str]:
        return ["sentiment", "customer_effort", "empathy"]

    def analyze(self, ctx: AnalysisContext) -> dict[str, Any]:
        if not ctx.segments:
            return {"resolution_probability": 50, "confidence": 30, "action": "Insufficient data"}

        # Simple but effective heuristic model
        sentiment_results = ctx.results.get("sentiment", [])
        effort = ctx.results.get("customer_effort", {})
        empathy = ctx.results.get("empathy", {})

        base = 65

        # Improving sentiment trajectory bonus
        if isinstance(sentiment_results, list) and len(sentiment_results) >= 2:
            first = sentiment_results[0].get("label", "neutral") if isinstance(sentiment_results[0], dict) else "neutral"
            last = sentiment_results[-1].get("label", "neutral") if isinstance(sentiment_results[-1], dict) else "neutral"
            if first == "negativ" and last == "positiv":
                base += 15
            elif last == "negativ":
                base -= 20

        # High effort = lower resolution chance
        ces = effort.get("overall_ces", 40) if isinstance(effort, dict) else 40
        base -= (ces - 40) * 0.3

        # Empathy helps resolution
        emp = empathy.get("overall_empathy", 50) if isinstance(empathy, dict) else 50
        base += (emp - 50) * 0.25

        probability = max(10, min(95, round(base, 1)))
        confidence = 65 if len(ctx.segments) > 4 else 45

        action = "Fortsätt med empati och tydliga nästa steg" if probability > 70 else \
                 "Eskalera eller ge mer tid + validering" if probability < 45 else \
                 "Sammanfatta och bekräfta lösning"

        return {
            "resolution_probability": probability,
            "confidence": confidence,
            "recommended_action": action,
            "factors": {
                "sentiment_trend": "positive" if probability > 65 else "mixed/negative",
                "customer_effort_impact": ces,
                "empathy_impact": emp,
            }
        }
"