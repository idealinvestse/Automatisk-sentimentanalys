"""Predictive analytics for Swedish call center conversations.

Detects:
    - Churn risk (customer likely to leave)
    - Escalation risk (issue likely to be escalated)
    - Customer satisfaction score prediction

Uses rule-based heuristics by default. Model-based prediction
requires training data (see src/finetune.py).

Usage:
    from src.predictive import RiskAnalyzer
    ra = RiskAnalyzer()
    churn_risk, escalation_risk = ra.analyze(sentiment_results, intent_results, segments)
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RiskAssessment:
    """Risk assessment for a single call."""

    churn_risk: float = 0.0  # 0-1
    escalation_risk: float = 0.0  # 0-1
    satisfaction_score: float = 0.5  # 0-1
    risk_factors: list[str] = field(default_factory=list)
    risk_level: str = "low"  # "low", "medium", "high", "critical"

    def to_dict(self) -> dict[str, Any]:
        return {
            "churn_risk": round(self.churn_risk, 3),
            "escalation_risk": round(self.escalation_risk, 3),
            "satisfaction_score": round(self.satisfaction_score, 3),
            "risk_factors": self.risk_factors,
            "risk_level": self.risk_level,
        }


class RiskAnalyzer:
    """Analyze churn and escalation risk for call center conversations.

    Args:
        backend: 'heuristic' (default) or 'model' (requires training).
        model_path: Path to trained risk model.
    """

    def __init__(
        self,
        backend: str = "heuristic",
        model_path: str | None = None,
    ) -> None:
        self.backend = backend
        self.model_path = model_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(
        self,
        sentiment_results: list[dict[str, Any]] | None = None,
        intent_results: list[tuple[str, float]] | None = None,
        segments: list[dict[str, Any]] | None = None,
    ) -> RiskAssessment:
        """Analyze risks for a conversation.

        Args:
            sentiment_results: Per-segment sentiment results.
            intent_results: Per-segment intent classifications.
            segments: ASR transcript segments.

        Returns:
            RiskAssessment with churn risk, escalation risk, and satisfaction score.
        """
        risk_factors: list[str] = []

        # --- Churn risk ---
        churn_score = self._compute_churn_risk(sentiment_results, intent_results, risk_factors)

        # --- Escalation risk ---
        escalation_score = self._compute_escalation_risk(
            sentiment_results, intent_results, risk_factors
        )

        # --- Satisfaction score ---
        satisfaction = self._compute_satisfaction(sentiment_results, intent_results)

        # --- Overall risk level ---
        max_risk = max(churn_score, escalation_score)
        if max_risk >= 0.7:
            level = "critical"
        elif max_risk >= 0.5:
            level = "high"
        elif max_risk >= 0.3:
            level = "medium"
        else:
            level = "low"

        return RiskAssessment(
            churn_risk=churn_score,
            escalation_risk=escalation_score,
            satisfaction_score=satisfaction,
            risk_factors=risk_factors,
            risk_level=level,
        )

    def analyze_batch(
        self,
        calls_data: list[dict[str, Any]],
    ) -> list[RiskAssessment]:
        """Analyze risks for multiple calls."""
        results: list[RiskAssessment] = []
        for call in calls_data:
            results.append(
                self.analyze(
                    sentiment_results=call.get("sentiment_results"),
                    intent_results=call.get("intent_results"),
                    segments=call.get("segments"),
                )
            )
        return results

    # ------------------------------------------------------------------
    # Risk computation
    # ------------------------------------------------------------------
    def _compute_churn_risk(
        self,
        sentiment_results: list[dict[str, Any]] | None,
        intent_results: list[tuple[str, float]] | None,
        risk_factors: list[str],
    ) -> float:
        """Compute churn risk score."""
        score = 0.0
        factor_count = 0

        # High negative sentiment → churn signal
        if sentiment_results:
            neg_count = sum(1 for r in sentiment_results if r.get("label") == "negativ")
            neg_ratio = neg_count / max(1, len(sentiment_results))
            if neg_ratio > 0.5:
                score += 0.7
                risk_factors.append(f"Hög andel negativitet ({neg_ratio:.0%})")
                factor_count += 1
            elif neg_ratio > 0.3:
                score += 0.4
                risk_factors.append(f"Förhöjd negativitet ({neg_ratio:.0%})")
                factor_count += 1

        # Cancellation intent → strong churn signal
        if intent_results:
            cancel_count = sum(1 for i, _ in intent_results if i == "cancellation")
            if cancel_count > 0:
                score += 0.8
                risk_factors.append("Kund uttrycker uppsägningsönskemål")
                factor_count += 1

            complaint_count = sum(1 for i, _ in intent_results if i == "complaint")
            if complaint_count >= 2:
                score += 0.5
                risk_factors.append(f"{complaint_count} klagomålssegment")
                factor_count += 1

        return min(1.0, score / max(1, factor_count)) if factor_count > 0 else 0.1

    def _compute_escalation_risk(
        self,
        sentiment_results: list[dict[str, Any]] | None,
        intent_results: list[tuple[str, float]] | None,
        risk_factors: list[str],
    ) -> float:
        """Compute escalation risk score."""
        score = 0.0
        factor_count = 0

        if intent_results:
            complaint_count = sum(1 for i, _ in intent_results if i == "complaint")
            if complaint_count >= 3:
                score += 0.9
                risk_factors.append("Flera klagomål – eskaleringsrisk")
                factor_count += 1
            elif complaint_count >= 1:
                score += 0.4
                factor_count += 1

            refund_count = sum(1 for i, _ in intent_results if i == "refund_request")
            if refund_count > 0:
                score += 0.3
                risk_factors.append("Återbetalningskrav")
                factor_count += 1

        # Long calls with negative sentiment
        if sentiment_results and len(sentiment_results) > 15:
            neg_ratio = sum(1 for r in sentiment_results if r.get("label") == "negativ") / len(
                sentiment_results
            )
            if neg_ratio > 0.4:
                score += 0.5
                risk_factors.append("Långt samtal med hög negativitet")
                factor_count += 1

        return min(1.0, score / max(1, factor_count)) if factor_count > 0 else 0.05

    def _compute_satisfaction(
        self,
        sentiment_results: list[dict[str, Any]] | None,
        intent_results: list[tuple[str, float]] | None,
    ) -> float:
        """Compute a satisfaction score (0-1)."""
        if not sentiment_results:
            return 0.5

        labels = [r.get("label", "neutral") for r in sentiment_results]
        counts = Counter(labels)
        total = max(1, len(labels))

        # Weighted score: positive=1, neutral=0.5, negative=0
        score = (counts.get("positiv", 0) * 1.0 + counts.get("neutral", 0) * 0.5) / total

        # Penalize complaints
        if intent_results:
            complaint_ratio = sum(1 for i, _ in intent_results if i == "complaint") / max(
                1, len(intent_results)
            )
            score = max(0.0, score - complaint_ratio * 0.3)

        return round(min(1.0, max(0.0, score)), 3)


__all__ = ["RiskAnalyzer", "RiskAssessment"]
