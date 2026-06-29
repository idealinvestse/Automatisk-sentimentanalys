"""Conversation Trajectory & Escalation (Task 2.3)."""

from __future__ import annotations

import logging
from typing import Any

from ..core.models import AnalysisContext
from .base import Analyzer
from .registry import register_analyzer

logger = logging.getLogger(__name__)

_ESCALATION_EMOTIONS = frozenset({"frustration", "ilska", "besvikelse", "oro"})


def _customer_speakers(role_result: dict[str, Any]) -> set[str]:
    roles = role_result.get("roles") if isinstance(role_result, dict) else {}
    if not isinstance(roles, dict):
        return set()
    return {sp for sp, role in roles.items() if str(role).lower() == "customer"}


def _is_customer_turn(seg: Any, customer_speakers: set[str]) -> bool:
    if not customer_speakers:
        return True
    speaker = getattr(seg, "speaker", None)
    return speaker in customer_speakers


@register_analyzer("trajectory")
class TrajectoryAnalyzer(Analyzer):
    @property
    def name(self) -> str:
        return "trajectory"

    @property
    def requires(self) -> list[str]:
        return ["sentiment", "emotion", "role"]

    def analyze(self, ctx: AnalysisContext) -> dict[str, Any]:
        sentiment_results = ctx.results.get("sentiment", []) or []
        emotion_results = ctx.results.get("emotion", []) or []
        role_result = ctx.results.get("role") or {}
        customer_speakers = _customer_speakers(role_result)

        customer_sentiment_trend: list[float] = []
        escalation_event_details: list[dict[str, Any]] = []
        peak_frustration_turn: int | None = None
        peak_frustration_score = -1.0
        prev_customer_score: float | None = None

        for i, seg in enumerate(ctx.segments or []):
            if not _is_customer_turn(seg, customer_speakers):
                continue

            sent = sentiment_results[i] if i < len(sentiment_results) else {}
            score = float(sent.get("score", 0)) if isinstance(sent, dict) else 0.0
            label = (
                str(sent.get("label", "neutral")).lower() if isinstance(sent, dict) else "neutral"
            )
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
            snippet = (getattr(seg, "text", None) or "")[:80]

            if primary in _ESCALATION_EMOTIONS:
                escalation_event_details.append(
                    {
                        "turn": i,
                        "type": "emotion",
                        "emotion": primary,
                        "evidence": snippet,
                    }
                )
                if emo_score > peak_frustration_score:
                    peak_frustration_score = emo_score
                    peak_frustration_turn = i

            if prev_customer_score is not None and score < prev_customer_score - 0.3:
                escalation_event_details.append(
                    {
                        "turn": i,
                        "type": "sentiment_drop",
                        "delta": round(score - prev_customer_score, 3),
                        "evidence": snippet,
                    }
                )
            prev_customer_score = score

        slope = 0.0
        if len(customer_sentiment_trend) > 1:
            slope = (customer_sentiment_trend[-1] - customer_sentiment_trend[0]) / max(
                1, len(customer_sentiment_trend) - 1
            )

        return {
            "customer_sentiment_slope": round(slope, 3),
            "escalation_events": len(escalation_event_details),
            "escalation_event_details": escalation_event_details,
            "peak_frustration_turn": peak_frustration_turn,
            "sentiment_trend": customer_sentiment_trend,
        }
