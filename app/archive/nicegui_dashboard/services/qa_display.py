"""QA score tier and CSS class helpers for dashboard display."""

from __future__ import annotations

from typing import Any


def _coerce_score(score: Any) -> float | None:
    if score is None or score == "—":
        return None
    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def qa_score_tier(score: Any) -> str:
    """Return high | mid | low | none for a QA score."""
    value = _coerce_score(score)
    if value is None:
        return "none"
    if value >= 80:
        return "high"
    if value >= 60:
        return "mid"
    return "low"


def qa_score_css_class(score: Any) -> str:
    """Quasar text color class for QA score display."""
    return {
        "high": "text-positive",
        "mid": "text-warning",
        "low": "text-negative",
        "none": "text-grey",
    }[qa_score_tier(score)]


def qa_chip_color(score: Any) -> str:
    """Chip color prop for call detail header."""
    return {
        "high": "positive",
        "mid": "warning",
        "low": "negative",
        "none": "grey",
    }[qa_score_tier(score)]
