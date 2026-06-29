"""Predictive / churn-escalation analyzer (registry adapter for RiskAnalyzer)."""

from __future__ import annotations

import logging
from typing import Any

from ..core.models import AnalysisContext
from ..predictive import RiskAnalyzer
from .base import Analyzer
from .intent_utils import intents_as_tuples
from .registry import register_analyzer

logger = logging.getLogger(__name__)


@register_analyzer("predictive")
class PredictiveAnalyzer(Analyzer):
    """Escalation/churn risk via :class:`~src.predictive.RiskAnalyzer` (numeric 0–1 scores)."""

    def __init__(self, backend: str = "heuristic") -> None:
        self.backend = backend
        self._analyzer: RiskAnalyzer | None = None

    @property
    def name(self) -> str:
        return "predictive"

    @property
    def requires(self) -> list[str]:
        return ["sentiment", "intent"]

    def _get_analyzer(self) -> RiskAnalyzer:
        if self._analyzer is None:
            self._analyzer = RiskAnalyzer(backend=self.backend)
        return self._analyzer

    def analyze(self, ctx: AnalysisContext) -> dict[str, Any]:
        if not ctx.segments:
            return RiskAnalyzer().analyze().to_dict()

        sentiment_results = ctx.results.get("sentiment")
        intent_results = intents_as_tuples(ctx.results.get("intent") or [])
        segments_dict = [s.to_dict() for s in ctx.segments]

        try:
            assessment = self._get_analyzer().analyze(
                sentiment_results=sentiment_results,
                intent_results=intent_results,
                segments=segments_dict,
            )
            out = assessment.to_dict()
            # Backward-compatible alias used by some dashboards
            out["recommended_action"] = (
                "Supervisor review" if assessment.risk_level in ("high", "critical") else None
            )
            return out
        except Exception as exc:
            logger.warning("PredictiveAnalyzer failed: %s", exc)
            return RiskAnalyzer().analyze().to_dict()
