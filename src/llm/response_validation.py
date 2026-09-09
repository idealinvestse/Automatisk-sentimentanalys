"""Domain validation for structured call-analysis responses."""

from __future__ import annotations

import re
from typing import Any

from ..core.errors import LLMError
from ..core.models import Segment

_TASK_FIELDS = {
    "trajectory": "trajectory",
    "refined_aspects": "refined_aspects",
    "root_cause": "root_cause",
    "actionable_summary": "actionable_summary",
    "agent_assessment": "agent_assessment",
    "agent_assessment_detailed": "agent_assessment",
    "emotion_trajectory": "emotion_trajectory",
}


def _normalize_evidence(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().casefold())


def _segment_texts(segments: list[dict[str, Any]] | list[Segment]) -> list[str]:
    values: list[str] = []
    for segment in segments:
        text = segment.get("text", "") if isinstance(segment, dict) else segment.text
        values.append(_normalize_evidence(str(text)))
    return values


def _evidence_items(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if isinstance(value.get("text"), str) and (
            "turn_index" in value or "segment_id" in value or "speaker_role" in value
        ):
            found.append(value)
        for child in value.values():
            found.extend(_evidence_items(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_evidence_items(child))
    return found


def validate_call_llm_response(
    result: dict[str, Any],
    *,
    tasks: list[str],
    segments: list[dict[str, Any]] | list[Segment],
) -> dict[str, Any]:
    """Require requested tasks and reject evidence that cannot be traced to a segment."""
    missing = sorted(
        {
            field
            for task in tasks
            if (field := _TASK_FIELDS.get(task)) is not None and field not in result
        }
    )
    if missing:
        raise LLMError(
            "LLM response omitted requested analysis tasks",
            error_code="llm_task_coverage_failed",
            details={"missing_fields": missing},
        )

    texts = _segment_texts(segments)
    invalid: list[dict[str, Any]] = []
    for item in _evidence_items(result):
        evidence = _normalize_evidence(str(item.get("text") or ""))
        segment_id = item.get("segment_id", item.get("turn_index"))
        if isinstance(segment_id, int) and 0 <= segment_id < len(texts):
            matched = bool(evidence and evidence in texts[segment_id])
        else:
            matches = [index for index, text in enumerate(texts) if evidence and evidence in text]
            matched = bool(matches)
            if matches:
                segment_id = matches[0]
                item["segment_id"] = segment_id
                item["turn_index"] = segment_id
        if not matched:
            invalid.append({"text": str(item.get("text") or "")[:120], "segment_id": segment_id})
        elif isinstance(segment_id, int):
            source = segments[segment_id]
            if isinstance(source, dict):
                item["start"] = source.get("start")
                item["end"] = source.get("end")
            else:
                item["start"] = getattr(source, "start", None)
                item["end"] = getattr(source, "end", None)
    if invalid:
        raise LLMError(
            "LLM response contained evidence not found in the transcript",
            error_code="llm_evidence_validation_failed",
            details={"invalid_evidence": invalid[:10], "invalid_count": len(invalid)},
        )
    return result
