"""Call summarization analyzer adapter."""

from __future__ import annotations

import logging
from typing import Any

from ..core.models import AnalysisContext
from ..summarizer import CallSummarizer, CallSummary
from .base import Analyzer
from .registry import register_analyzer

logger = logging.getLogger(__name__)


@register_analyzer("summary")
class SummaryAnalyzer(Analyzer):
    """Analyzer that generates summaries and action items for a call."""

    def __init__(self) -> None:
        self._summarizer: CallSummarizer | None = None

    @property
    def name(self) -> str:
        return "summary"

    @property
    def requires(self) -> list[str]:
        return ["sentiment", "intent"]

    def _get_summarizer(self) -> CallSummarizer:
        if self._summarizer is None:
            self._summarizer = CallSummarizer()
        return self._summarizer

    def analyze(self, ctx: AnalysisContext) -> dict[str, Any]:
        if not ctx.segments:
            return CallSummary().to_dict()

        segments_dict = [s.to_dict() for s in ctx.segments]
        sentiment_results = ctx.results.get("sentiment")
        intent_results = ctx.results.get("intent")

        try:
            summarizer = self._get_summarizer()
            report = summarizer.summarize(
                segments_dict,
                sentiment_results=sentiment_results,
                intent_results=intent_results,
            )
            return report.to_dict()
        except Exception as e:
            logger.error("Summarization failed in adapter: %s", e)
            return CallSummary().to_dict()
