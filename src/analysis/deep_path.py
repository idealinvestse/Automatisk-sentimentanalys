"""Deep-path analyzer selection — which local analyzers LLM supersedes."""

from __future__ import annotations

LLM_SUPERSEDED_ANALYZERS: frozenset[str] = frozenset(
    {
        "empathy",
        "trajectory",
        "insights",
        "root_cause",
        "actionable_coaching",
    }
)


def filter_superseded(selected: list[str] | None, *, skip: bool) -> list[str] | None:
    """Remove LLM-superseded analyzers from *selected* when *skip* is True."""
    if not skip or selected is None:
        return selected
    filtered = [name for name in selected if name not in LLM_SUPERSEDED_ANALYZERS]
    skipped = [name for name in selected if name in LLM_SUPERSEDED_ANALYZERS]
    if skipped:
        import logging

        logging.getLogger(__name__).info(
            "Skipping local analyzers superseded by LLM deep path: %s", skipped
        )
    return filtered
