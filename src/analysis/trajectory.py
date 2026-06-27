"""Conversation Trajectory & Escalation (Task 2.3)."""

from __future__ import annotations

import logging
from typing import Any

from ..core.models import AnalysisContext
from .base import Analyzer
from .registry import register_analyzer

logger = logging.getLogger(__name__)

_ESCALATION_EMOTIONS = frozenset({"frustration", "ilska", "besvikelse", "oro"})


@register_analyzer("trajectory")
class TrajectoryAnalyzer(Analyzer):
    @property
    def name(self) -> str:
        return "trajectory"

    @property
    def requires(self) -> list[str]:
        return ["sentiment", "emotion"]

    def analyze(self, ctx: AnalysisContext) -> dict[str, Any]:
        sentiment_results = ctx.results.get("sentiment", []) or []
        emotion_results = ctx.results.get("emotion", []) or []

        customer_sentiment_trend: list[float] = []
        escalation_events = 0
        peak_frustration_turn: int | None = None
        peak_frustration_score = -1.0

        for i, seg in enumerate(ctx.segments or []):
            sent = sentiment_results[i] if i < len(sentiment_results) else {}
            score = float(sent.get("score", 0)) if isinstance(sent, dict) else 0.0
            label = str(sent.get("label", "neutral")).lower() if isinstance(sent, dict) else "neutral"
            if label in ("negativ", "negative"):
                score = -abs(score)
            elif label in ("positiv", "positive"):
                score = abs(score)
            customer_sentiment_trend.append(score)

            emo = emotion_results[i] if i < len(emotion_results) else {}
            primary = str(emo.get("primary", "neutral")) if isinstance(emo, dict) else "neutral"
            emo_score = 0.0
            if isinstance(emo, dict) and emo.get("scores"):
                emo_score = float(max(emo["scores"].values()))
            if primary in _ESCALATION_EMOTIONS:
                escalation_events += 1
                if emo_score > peak_frustration_score:
                    peak_frustration_score = emo_score
                    peak_frustration_turn = i

            if i > 0 and customer_sentiment_trend[i] < customer_sentiment_trend[i - 1] - 0.3:
                escalation_events += 1

        slope = 0.0
        if len(customer_sentiment_trend) > 1:
            slope = (customer_sentiment_trend[-1] - customer_sentiment_trend[0]) / max(
                1, len(customer_sentiment_trend) - 1
            )

        return {
            "customer_sentiment_slope": round(slope, 3),
            "escalation_events": escalation_events,
            "peak_frustration_turn": peak_frustration_turn,
            "sentiment_trend": customer_sentiment_trend,
        }