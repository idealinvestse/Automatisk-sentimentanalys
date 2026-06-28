"""ActionableCoachingInsight – concrete coaching recommendations from call signals."""

from __future__ import annotations

import logging
from typing import Any

from ..core.models import AnalysisContext
from .base import Analyzer
from .registry import register_analyzer

logger = logging.getLogger(__name__)

_COACHING_RULES: list[tuple[str, str, list[str]]] = [
    ("low_empathy", "Empatin under 50 – träna validerande fraser tidigt i samtalet.", ["empathy"]),
    ("high_effort", "Hög kundinsats (CES) – förenkla språk och bekräfta förståelse.", ["customer_effort"]),
    ("compliance_flag", "Compliance-risk flaggad – granska löften och dataförfrågningar.", ["compliance_risk"]),
    ("negative_trajectory", "Försämrad sentimentkurva – pausa script och validera känslor.", ["trajectory"]),
]


@register_analyzer("actionable_coaching")
class ActionableCoachingAnalyzer(Analyzer):
    """Synthesizes prior analyzer outputs into prioritized coaching actions."""

    @property
    def name(self) -> str:
        return "actionable_coaching"

    @property
    def requires(self) -> list[str]:
        return ["empathy", "customer_effort", "trajectory", "compliance_risk"]

    def analyze(self, ctx: AnalysisContext) -> dict[str, Any]:
        empathy = ctx.results.get("empathy") or {}
        effort = ctx.results.get("customer_effort") or {}
        trajectory = ctx.results.get("trajectory") or {}
        compliance = ctx.results.get("compliance_risk") or {}

        insights: list[dict[str, Any]] = []

        emp_score = float(empathy.get("overall_empathy", 50)) if isinstance(empathy, dict) else 50
        if emp_score < 50:
            insights.append(self._item("low_empathy", "high", emp_score))

        ces = float(effort.get("overall_ces", 0)) if isinstance(effort, dict) else 0
        if ces > 45:
            insights.append(self._item("high_effort", "medium", ces))

        if isinstance(compliance, dict) and compliance.get("overall_risk_level") in ("medium", "high"):
            insights.append(self._item("compliance_flag", "high", compliance.get("flagged_segments", [])))

        slope = float(trajectory.get("customer_sentiment_slope", 0)) if isinstance(trajectory, dict) else 0
        if slope < -0.05:
            insights.append(self._item("negative_trajectory", "medium", slope))

        priority_order = {"high": 0, "medium": 1, "low": 2}
        insights.sort(key=lambda x: priority_order.get(x["priority"], 9))

        return {
            "coaching_insights": insights,
            "top_recommendation": insights[0]["recommendation"] if insights else None,
            "insight_count": len(insights),
        }

    def _item(self, rule_id: str, priority: str, evidence: Any) -> dict[str, Any]:
        for rid, rec, _ in _COACHING_RULES:
            if rid == rule_id:
                return {
                    "rule_id": rule_id,
                    "priority": priority,
                    "recommendation": rec,
                    "evidence": evidence,
                }
        return {"rule_id": rule_id, "priority": priority, "recommendation": rule_id, "evidence": evidence}