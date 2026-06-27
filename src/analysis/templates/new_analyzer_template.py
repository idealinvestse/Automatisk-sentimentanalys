"""Template for new analyzers – copy to src/analysis/your_analyzer.py and customize."""

from __future__ import annotations

from typing import Any

from ..base import Analyzer
from ...core.models import AnalysisContext
from ..registry import register_analyzer


@register_analyzer("your_analyzer")
class YourAnalyzer(Analyzer):
    """Short description of what this analyzer produces for call center QA/coaching."""

    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "your_analyzer"

    @property
    def requires(self) -> list[str]:
        # List analyzers that must run first, e.g. ["sentiment", "role"]
        return []

    def analyze(self, ctx: AnalysisContext) -> dict[str, Any]:
        if not ctx.segments:
            return {"status": "empty"}

        # Read prior results: ctx.results.get("sentiment")
        return {
            "status": "ok",
            "segment_count": len(ctx.segments),
        }