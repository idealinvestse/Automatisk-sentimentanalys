"""Predictive risk analyzer adapter."""

from __future__ import annotations

import logging
from typing import Any

from ..core.models import AnalysisContext
from ..predictive import RiskAnalyzer, RiskAssessment
from .base import Analyzer
from .registry import register_analyzer

logger = logging.getLogger(__name__)


@register_analyzer("predictive")
class PredictiveAnalyzer(Analyzer):
    """Analyzer that runs predictive churn/escalation risk analysis."""

    def __init__(self) -> None:
        self._analyzer: RiskAnalyzer | None = None

    @property
    def name(self) -> str:
        return "predictive"

    @property
    def requires(self) -> list[str]:
        return ["sentiment", "intent"]

    def _get_analyzer(self) -> RiskAnalyzer:
        if self._analyzer is None:
            self._analyzer = RiskAnalyzer()
        return self._analyzer

    def analyze(self, ctx: AnalysisContext) -> dict[str, Any]:
        sentiment_results = ctx.results.get("sentiment")
        intent_results = ctx.results.get("intent")

        try:
            analyzer = self._get_analyzer()
            report = analyzer.analyze(
                sentiment_results=sentiment_results,
                intent_results=intent_results,
            )
            return report.to_dict()
        except Exception as e:
            logger.error("Risk analysis failed in adapter: %s", e)
            return RiskAssessment().to_dict()
