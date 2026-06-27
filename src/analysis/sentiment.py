"""Sentiment analysis analyzer adapter."""

from __future__ import annotations

import logging
from typing import Any

from ..core.config import DEFAULT_SENTIMENT_MODEL
from ..core.models import AnalysisContext
from ..sentiment import SentimentPipeline
from .base import Analyzer
from .registry import register_analyzer

logger = logging.getLogger(__name__)


@register_analyzer("sentiment")
class SentimentAnalyzer(Analyzer):
    """Analyzer that runs sentiment analysis on conversation segments."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        return_all_scores: bool = False,
    ) -> None:
        self.model_name = model_name or DEFAULT_SENTIMENT_MODEL
        self.device = device or "auto"
        self.return_all_scores = return_all_scores
        self._pipeline: SentimentPipeline | None = None

    @property
    def name(self) -> str:
        return "sentiment"

    @property
    def requires(self) -> list[str]:
        return []

    def _get_pipeline(self) -> SentimentPipeline:
        if self._pipeline is None:
            from .resources import get_pool

            self._pipeline = get_pool().get_sentiment_pipeline(
                model_name=self.model_name,
                device=self.device,
                return_all_scores=self.return_all_scores,
            )
        return self._pipeline

    def analyze(self, ctx: AnalysisContext) -> list[dict[str, Any]]:
        if not ctx.segments:
            return []

        texts = [s.text for s in ctx.segments]
        try:
            pipeline = self._get_pipeline()
            results = pipeline.analyze(
                texts,
                normalize=True,
                return_all_scores=self.return_all_scores,
            )
            return results
        except Exception as e:
            logger.error("Sentiment analysis failed in adapter: %s", e)
            # Fallback
            fallback_label = "neutral"
            if self.return_all_scores:
                return [
                    [
                        {"label": "negativ", "score": 0.0},
                        {"label": "neutral", "score": 1.0},
                        {"label": "positiv", "score": 0.0},
                    ]
                    for _ in texts
                ]
            return [{"label": fallback_label, "score": 0.0} for _ in texts]
