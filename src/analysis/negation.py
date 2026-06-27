"""Negation detection analyzer adapter for the registry dependency graph."""

from __future__ import annotations

import logging
from typing import Any

from ..core.models import AnalysisContext
from ..negation import detect_negation, detect_negation_with_position
from .base import Analyzer
from .registry import register_analyzer

logger = logging.getLogger(__name__)


@register_analyzer("negation")
class NegationAnalyzer(Analyzer):
    """Detect Swedish negation markers per segment (feeds empathy and sentiment QA)."""

    @property
    def name(self) -> str:
        return "negation"

    @property
    def requires(self) -> list[str]:
        return []

    def analyze(self, ctx: AnalysisContext) -> list[dict[str, Any]]:
        if not ctx.segments:
            return []

        out: list[dict[str, Any]] = []
        for seg in ctx.segments:
            text = seg.text or ""
            positions = detect_negation_with_position(text)
            out.append(
                {
                    "has_negation": bool(positions) or detect_negation(text),
                    "negation_count": len(positions),
                    "positions": [{"index": idx, "type": ntype} for idx, ntype in positions],
                    "speaker": getattr(seg, "speaker", None),
                }
            )
        return out