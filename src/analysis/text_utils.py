"""Shared text helpers for analyzer adapters."""

from __future__ import annotations

from ..core.models import AnalysisContext


def segment_analysis_text(ctx: AnalysisContext, index: int) -> str:
    """Return normalized segment text when spoken_normalizer has run, else raw transcript."""
    norm = ctx.results.get("spoken_normalizer")
    if isinstance(norm, list) and index < len(norm):
        item = norm[index]
        if isinstance(item, dict) and item.get("normalized"):
            return str(item["normalized"])
    segments = ctx.segments or []
    if index < len(segments):
        return segments[index].text or ""
    return ""
