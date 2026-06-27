"""Predictive / churn-escalation analyzer (heuristic stub, registry-compatible)."""

from __future__ import annotations

import logging
from typing import Any

from ..core.models import AnalysisContext
from .base import Analyzer
from .registry import register_analyzer

logger = logging.getLogger(__name__)

ESCALATION_MARKERS = ["eskalera", "chef", "klaga", "konsumentverket", "advokat", "aldrig mer"]


@register_analyzer("predictive")
class PredictiveAnalyzer(Analyzer):
    """Lightweight escalation/churn risk heuristic for backward-compatible ``risks`` mapping."""

    @property
    def name(self) -> str:
        return "predictive"

    @property
    def requires(self) -> list[str]:
        return ["sentiment", "trajectory"]

    def analyze(self, ctx: AnalysisContext) -> dict[str, Any]:
        if not ctx.segments:
            return {"escalation_risk": "low", "churn_risk": "low", "signals": []}

        text_blob = " ".join((s.text or "").lower() for s in ctx.segments)
        signals = [m for m in ESCALATION_MARKERS if m in text_blob]
        trajectory = ctx.results.get("trajectory", {})
        slope = 0.0
        if isinstance(trajectory, dict):
            slope = float(trajectory.get("customer_sentiment_slope", 0.0))

        risk_score = len(signals) * 15 + (10 if slope < -0.1 else 0)
        level = "high" if risk_score >= 30 else ("medium" if risk_score >= 15 else "low")

        return {
            "escalation_risk": level,
            "churn_risk": level if level != "low" else "low",
            "signals": signals,
            "sentiment_slope": slope,
            "recommended_action": "Supervisor review" if level == "high" else None,
        }