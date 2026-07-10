"""Shared EvidenceSpan contract for analyzers and LLM overrides.

Canonical quote/evidence unit used by aspect ABSA, compliance, provenance,
and LLM claim charts. Re-exports the Pydantic model from ``src.llm.schemas``
after extending it with segment timing fields (backward compatible).
"""

from __future__ import annotations

from typing import Any

from ..llm.schemas import EvidenceSpan

__all__ = ["EvidenceSpan", "make_evidence_span", "evidence_to_dict"]


def make_evidence_span(
    text: str,
    *,
    speaker_role: str | None = None,
    turn_index: int | None = None,
    segment_id: int | None = None,
    start: float | None = None,
    end: float | None = None,
) -> EvidenceSpan:
    """Build a validated EvidenceSpan (quote = ``text``)."""
    return EvidenceSpan(
        text=(text or "")[:500],
        speaker_role=speaker_role,
        turn_index=turn_index,
        segment_id=segment_id,
        start=start,
        end=end,
    )


def evidence_to_dict(span: EvidenceSpan | dict[str, Any]) -> dict[str, Any]:
    """Serialize an EvidenceSpan (or already-dict) for analyzer JSON results."""
    if isinstance(span, EvidenceSpan):
        return span.model_dump()
    return dict(span)
