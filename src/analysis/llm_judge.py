"""Lightweight LLM-Judge fallback for low-confidence cases (Task 2.4).

In real deployment this would call a local quantized LLM (e.g. via api.intelliserve.se or transformers).
Here we provide a stub that logs usage and falls back to existing heuristics.
"""

from __future__ import annotations

import logging
from typing import Any

from ..core.models import AnalysisContext
from .base import Analyzer
from .registry import register_analyzer

logger = logging.getLogger(__name__)


@register_analyzer("llm_judge")
class LLMJudgeAnalyzer(Analyzer):
    @property
    def name(self) -> str:
        return "llm_judge"

    @property
    def requires(self) -> list[str]:
        return ["sentiment"]

    def analyze(self, ctx: AnalysisContext) -> list[dict[str, Any]]:
        # Stub: in production route low-conf segments here
        logger.info("LLM-Judge stub invoked (no real LLM call in this build)")
        # Return empty or pass-through
        return []
