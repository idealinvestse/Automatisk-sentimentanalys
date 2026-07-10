"""Deep-path analyzer selection, honest degradation, and supersession helpers."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

LLM_SUPERSEDED_ANALYZERS: frozenset[str] = frozenset(
    {
        "empathy",
        "trajectory",
        "insights",
        "root_cause",
        "actionable_coaching",
    }
)

# Fields that must not pretend to be deep-path quality when LLM is off.
DEEP_PATH_REQUIRED_FIELDS: frozenset[str] = frozenset(LLM_SUPERSEDED_ANALYZERS)


def unavailable_payload(analyzer_name: str, *, reason: str = "requires_deep_path") -> dict[str, Any]:
    """Explicit null/unavailable marker — prefer honesty over quality-2 heuristics."""
    return {
        "status": "unavailable",
        "reason": reason,
        "analyzer": analyzer_name,
        "message": "Kräver deep path (LLM). Lokal heuristik körs inte (ärlig degradering).",
        "value": None,
    }


def is_unavailable(result: Any) -> bool:
    return isinstance(result, dict) and result.get("status") == "unavailable"


def filter_superseded(selected: list[str] | None, *, skip: bool) -> list[str] | None:
    """Remove LLM-superseded analyzers from *selected* when *skip* is True."""
    if not skip or selected is None:
        return selected
    filtered = [name for name in selected if name not in LLM_SUPERSEDED_ANALYZERS]
    skipped = [name for name in selected if name in LLM_SUPERSEDED_ANALYZERS]
    if skipped:
        logger.info("Skipping local analyzers superseded by LLM deep path: %s", skipped)
    return filtered


def filter_honest_degradation(
    selected: list[str] | None,
    *,
    deep_path_active: bool,
    allow_heuristic_superseded: bool = False,
) -> list[str] | None:
    """When deep path is off, drop superseded analyzers unless heuristics are explicitly allowed.

    Prevents quality-2 coaching/root_cause/empathy from appearing as if they were deep analysis.
    """
    if selected is None:
        return selected
    if deep_path_active or allow_heuristic_superseded:
        return selected
    filtered = [name for name in selected if name not in LLM_SUPERSEDED_ANALYZERS]
    dropped = [name for name in selected if name in LLM_SUPERSEDED_ANALYZERS]
    if dropped:
        logger.info(
            "Honest degradation: skipping superseded analyzers without deep path: %s",
            dropped,
        )
    return filtered


def inject_unavailable_markers(
    results: dict[str, Any],
    *,
    deep_path_active: bool,
    llm_used: bool = False,
) -> dict[str, Any]:
    """Ensure superseded fields are explicit unavailable when deep path did not produce them."""
    if deep_path_active and llm_used:
        return results
    for name in DEEP_PATH_REQUIRED_FIELDS:
        existing = results.get(name)
        if existing is None or is_unavailable(existing):
            results[name] = unavailable_payload(name)
        # If a heuristic somehow ran, replace with unavailable (null före bluff)
        elif not deep_path_active and not allow_keep_heuristic(existing):
            results[name] = unavailable_payload(name, reason="requires_deep_path")
    results.setdefault(
        "degradation",
        {
            "mode": "honest" if not (deep_path_active and llm_used) else "none",
            "deep_path_active": deep_path_active,
            "llm_used": llm_used,
        },
    )
    return results


def allow_keep_heuristic(_existing: Any) -> bool:
    """Hook for tests/opt-in; default never keep heuristic superseded output."""
    return False


def log_supersession(field: str, local_source: str | None, reason: str) -> None:
    logger.info(
        "LLM supersession | field=%s local_source=%s reason=%s",
        field,
        local_source,
        reason,
    )
