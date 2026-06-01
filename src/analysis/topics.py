"""Topic modeling analyzer adapter."""

from __future__ import annotations

import logging
from typing import Any

from ..core.models import AnalysisContext
from ..topic_modeling import TopicModeler, TopicReport
from .base import Analyzer
from .registry import register_analyzer

logger = logging.getLogger(__name__)


@register_analyzer("topics")
class TopicAnalyzer(Analyzer):
    """Analyzer that extracts topics from conversation segments."""

    def __init__(self) -> None:
        self._modeler: TopicModeler | None = None

    @property
    def name(self) -> str:
        return "topics"

    @property
    def requires(self) -> list[str]:
        # Topic modeling can benefit from having sentiment results
        return ["sentiment"]

    def _get_modeler(self) -> TopicModeler:
        if self._modeler is None:
            self._modeler = TopicModeler()
        return self._modeler

    def analyze(self, ctx: AnalysisContext) -> dict[str, Any]:
        if not ctx.segments:
            return TopicReport().to_dict()

        # TopicModeler expects a list of dict segments with 'text' and 'speaker' keys
        segments_dict = [s.to_dict() for s in ctx.segments]
        sentiment_results = ctx.results.get("sentiment")

        try:
            modeler = self._get_modeler()
            report = modeler.extract_topics(segments_dict, sentiment_results=sentiment_results)
            return report.to_dict()
        except Exception as e:
            logger.error("Topic modeling failed in adapter: %s", e)
            return TopicReport().to_dict()
