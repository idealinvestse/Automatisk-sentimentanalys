"""Shared helpers for normalizing intent analyzer output."""

from __future__ import annotations

from typing import Any


def intent_label(item: Any) -> str:
    """Extract intent label from dict or legacy tuple output."""
    if isinstance(item, dict):
        return str(item.get("intent", "other"))
    if isinstance(item, list | tuple) and item:
        return str(item[0])
    return "other"


def intent_confidence(item: Any) -> float:
    """Extract confidence from dict or legacy tuple output."""
    if isinstance(item, dict):
        return float(item.get("confidence", 0.0))
    if isinstance(item, list | tuple) and len(item) > 1:
        return float(item[1])
    return 0.0


def intents_as_tuples(items: list[Any]) -> list[tuple[str, float]]:
    """Convert analyzer output to legacy tuple form for CallAnalysisReport."""
    return [(intent_label(i), intent_confidence(i)) for i in items]


def intents_as_dicts(items: list[Any]) -> list[dict[str, Any]]:
    """Normalize intent results to dict form."""
    out: list[dict[str, Any]] = []
    for item in items:
        out.append({"intent": intent_label(item), "confidence": round(intent_confidence(item), 4)})
    return out
