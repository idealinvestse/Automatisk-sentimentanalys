"""Insights engine analyzer adapter."""

from __future__ import annotations

import logging
from typing import Any

from ..core.models import AnalysisContext
from ..insights import InsightsEngine, InsightsReport
from .base import Analyzer
from .registry import register_analyzer

logger = logging.getLogger(__name__)


@register_analyzer("insights")
class InsightsAnalyzer(Analyzer):
    """Analyzer that detects root causes and performance metrics for a call."""

    def __init__(self) -> None:
        self._engine: InsightsEngine | None = None

    @property
    def name(self) -> str:
        return "insights"

    @property
    def requires(self) -> list[str]:
        return ["sentiment", "intent", "topics"]

    def _get_engine(self) -> InsightsEngine:
        if self._engine is None:
            self._engine = InsightsEngine()
        return self._engine

    def analyze(self, ctx: AnalysisContext) -> dict[str, Any]:
        if not ctx.segments:
            return InsightsReport().to_dict()

        segments_dict = [s.to_dict() for s in ctx.segments]
        sentiment_results = ctx.results.get("sentiment")
        intent_results = ctx.results.get("intent")
        topics_result = ctx.results.get("topics", {})
        topics_list = topics_result.get("topics") if isinstance(topics_result, dict) else None

        try:
            engine = self._get_engine()
            report = engine.analyze(
                segments_dict,
                sentiment_results=sentiment_results,
                intent_results=intent_results,
                topics=topics_list,
            )
            return report.to_dict()
        except Exception as e:
            logger.error("Insights generation failed in adapter: %s", e)
            return InsightsReport().to_dict()
