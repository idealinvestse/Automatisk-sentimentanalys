"""Intent classification analyzer adapter."""

from __future__ import annotations

import logging

from ..core.models import AnalysisContext
from ..intent import IntentClassifier
from .base import Analyzer
from .registry import register_analyzer

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
            self._classifier = IntentClassifier(backend=self.backend)
        return self._classifier

    def analyze(self, ctx: AnalysisContext) -> list[tuple[str, float]]:
        if not ctx.segments:
            return []

        texts = [s.text for s in ctx.segments]
        try:
            classifier = self._get_classifier()
            results = classifier.classify_batch(texts)
            return results
        except Exception as e:
            logger.error("Intent classification failed in adapter: %s", e)
            return [("other", 0.0) for _ in texts]
