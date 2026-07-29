"""Override provenance and channel-diversity policy for hybrid LLM merges."""

from __future__ import annotations

import logging
from typing import Any

from ..llm.schemas import EvidenceSpan, OverrideProvenance
from .deep_path import log_supersession
from .evidence import make_evidence_span

logger = logging.getLogger(__name__)


def check_emotion_channel_diversity(
    local_emotion: Any,
    llm_emotion_trajectory: Any,
) -> tuple[bool, str]:
    """Keyword-emotion and LLM-emotion must not share the same sole lexical failure mode.

    Returns (ok, notes). Conservative: if both empty or both present with distinct shapes → ok.
    """
    local_has = bool(local_emotion)
    llm_has = bool(llm_emotion_trajectory)
    if local_has and llm_has:
        return (
            True,
            "Local emotion sensor and LLM emotion_trajectory both present (diverse channels).",
        )
    if not local_has and llm_has:
        return True, "LLM-only emotion channel (no local keyword emotion)."
    if local_has and not llm_has:
        return False, (
            "Diversity policy: local keyword/sentiment emotion without LLM channel — "
            "do not treat as deep-path emotion override."
        )
    return True, "No emotion channels active."


def build_override_provenance(
    results: dict[str, Any],
    llm_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build override_provenance entries for fields the LLM actually supplied."""
    if not llm_result or not llm_result.get("meta", {}).get("llm_used"):
        return []

    provenance: list[OverrideProvenance] = []
    diversity_ok, diversity_notes = check_emotion_channel_diversity(
        results.get("emotion"),
        llm_result.get("emotion_trajectory"),
    )

    def _spans_from_llm(field_val: Any) -> list[EvidenceSpan]:
        spans: list[EvidenceSpan] = []
        if isinstance(field_val, dict):
            raw = field_val.get("evidence_spans") or field_val.get("evidence") or []
            if isinstance(raw, list):
                for item in raw[:5]:
                    if isinstance(item, dict) and item.get("text"):
                        spans.append(
                            make_evidence_span(
                                str(item["text"]),
                                speaker_role=item.get("speaker_role"),
                                turn_index=item.get("turn_index"),
                                segment_id=item.get("segment_id"),
                                start=item.get("start"),
                                end=item.get("end"),
                            )
                        )
                    elif isinstance(item, str):
                        spans.append(make_evidence_span(item))
        return spans

    checks: list[tuple[str, str | None, Any]] = [
        ("agent_assessment", "agent_assessment_local", llm_result.get("agent_assessment")),
        ("trajectory", "trajectory", llm_result.get("trajectory")),
        ("root_cause", "root_cause", llm_result.get("root_cause")),
        ("refined_aspects", "aspect", llm_result.get("refined_aspects")),
        ("actionable_summary", "insights", llm_result.get("actionable_summary")),
    ]

    for field, local_source, llm_val in checks:
        if not llm_val:
            continue
        if local_source and results.get(local_source) is None and field != "refined_aspects":
            # Still record override when LLM fills a deep-path-only field
            pass
        entry = OverrideProvenance(
            field=field,
            local_source=local_source,
            reason="deep_path_holistic",
            evidence_spans=_spans_from_llm(llm_val),
            channel_diversity_ok=(
                diversity_ok if field in {"agent_assessment", "trajectory"} else True
            ),
            notes=diversity_notes if field in {"agent_assessment", "trajectory"} else None,
        )
        provenance.append(entry)
        log_supersession(field, local_source, "deep_path_holistic")

    return [p.model_dump() for p in provenance]


def apply_llm_overrides_with_provenance(
    results: dict[str, Any],
    llm_result: dict[str, Any],
) -> dict[str, Any]:
    """Merge LLM fields into results and attach override_provenance."""
    if not llm_result:
        return results

    provenance = build_override_provenance(results, llm_result)
    if provenance:
        llm_result = {**llm_result, "override_provenance": provenance}
        results["override_provenance"] = provenance
        results["llm"] = llm_result

    llm_assess = llm_result.get("agent_assessment")
    if isinstance(llm_assess, dict) and llm_assess.get("empathy_score") is not None:
        results["agent_assessment"] = {
            **llm_assess,
            "override_provenance": [p for p in provenance if p.get("field") == "agent_assessment"],
        }

    if llm_result.get("trajectory"):
        results["trajectory"] = llm_result["trajectory"]
    if llm_result.get("root_cause"):
        results["root_cause"] = llm_result["root_cause"]

    return results
