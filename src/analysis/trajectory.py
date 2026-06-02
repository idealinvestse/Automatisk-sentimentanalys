"""Conversation Trajectory & Escalation (Task 2.3)."""

from __future__ import annotations

import logging
from typing import Any

from ..core.models import AnalysisContext
from .base import Analyzer
from .registry import register_analyzer

logger = logging.getLogger(__name__)


@register_analyzer("trajectory")
class TrajectoryAnalyzer(Analyzer):
    @property
    def name(self) -> str:
        return "trajectory"

    @property
    def requires(self) -> list[str]:
        return ["sentiment", "emotion"]  # ideally role too

    def analyze(self, ctx: AnalysisContext) -> dict[str, Any]:
        customer_sentiment_trend = []
        escalation_events = 0
        for i, seg in enumerate(ctx.segments or []):
            # simplistic: use previous results if present
            sent = ctx.results.get("sentiment", [{}])[i] if i < len(ctx.results.get("sentiment", [])) else {}
            score = sent.get("score", 0) if isinstance(sent, dict) else 0
            customer_sentiment_trend.append(score)
            if i > 0 and customer_sentiment_trend[i] < customer_sentiment_trend[i-1] - 0.3:
                escalation_events += 1
        slope = 0.0
        if len(customer_sentiment_trend) > 1:
            slope = (customer_sentiment_trend[-1] - customer_sentiment_trend[0]) / max(1, len(customer_sentiment_trend)-1)
        return {
            "customer_sentiment_slope": round(slope, 3),
            "escalation_events": escalation_events,
            "peak_frustration_turn": None,  # can be enhanced with emotion results
        }
