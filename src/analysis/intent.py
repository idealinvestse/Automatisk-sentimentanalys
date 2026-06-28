"""Intent classification analyzer adapter."""

from __future__ import annotations

import logging
from typing import Any

from ..core.models import AnalysisContext
from ..intent import IntentClassifier
from .base import Analyzer
from .registry import register_analyzer
from .text_utils import segment_analysis_text

logger = logging.getLogger(__name__)


@register_analyzer("intent")
class IntentAnalyzer(Analyzer):
    """Analyzer that classifies the intent of conversation segments."""

    def __init__(self, backend: str = "heuristic") -> None:
        self.backend = backend
        self._classifier: IntentClassifier | None = None

    @property
    def name(self) -> str:
        return "intent"

    @property
    def requires(self) -> list[str]:
        return []

    def _get_classifier(self) -> IntentClassifier:
        if self._classifier is None:
            from .resources import get_pool

            self._classifier = get_pool().get_intent_classifier(backend=self.backend)
        return self._classifier

    def analyze(self, ctx: AnalysisContext) -> list[dict[str, Any]]:
        if not ctx.segments:
            return []

        texts = [segment_analysis_text(ctx, i) for i in range(len(ctx.segments))]
        try:
            classifier = self._get_classifier()
            results = classifier.classify_batch(texts)
            return [
                {"intent": label, "confidence": round(float(conf), 4)}
                for label, conf in results
            ]
        except Exception as e:
            logger.error("Intent classification failed in adapter: %s", e)
            return [{"intent": "other", "confidence": 0.0} for _ in texts]
