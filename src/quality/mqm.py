"""Quality OS scaffolding: MQM-like typology + preference gate hooks.

Full annotated corpus may be unavailable — schemas and evaluate hooks only.
Integrate real labels via DATA-01 (`scripts/import_domain_corpus.py`).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

MqmErrorType = Literal[
    "aspect_wrong",
    "aspect_missed",
    "intent_wrong",
    "emotion_wrong",
    "span_mismatch",
    "coaching_hallucination",
    "root_cause_wrong",
    "other",
]


class MqmError(BaseModel):
    """Single MQM-like annotation error on a call analysis output."""

    model_config = ConfigDict(extra="forbid")

    error_type: MqmErrorType
    severity: Literal["minor", "major", "critical"] = "major"
    analyzer: str | None = None
    span_text: str | None = Field(None, description="Evidence span related to the error")
    segment_id: int | None = None
    annotator_note: str | None = None


class MqmAnnotation(BaseModel):
    """Per-call MQM annotation record (corpus row)."""

    model_config = ConfigDict(extra="forbid")

    call_id: str
    errors: list[MqmError] = Field(default_factory=list)
    overall_quality: float = Field(ge=0.0, le=1.0, default=0.5)
    annotator: str | None = None


class PreferencePair(BaseModel):
    """Human preference between two deep-path / coaching outputs (release gate)."""

    model_config = ConfigDict(extra="forbid")

    call_id: str
    output_a_id: str
    output_b_id: str
    preferred: Literal["a", "b", "tie"]
    criterion: str = Field(
        "coaching_quality",
        description="What the rater preferred (coaching_quality, evidence, root_cause, ...)",
    )
    rater: str | None = None
    notes: str | None = None


class PreferenceGateResult(BaseModel):
    """Aggregate preference-gate decision for a model/config release."""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    n_pairs: int = 0
    win_rate_a: float | None = None
    min_win_rate: float = 0.55
    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


def evaluate_preference_gate(
    pairs: list[PreferencePair] | list[dict[str, Any]],
    *,
    candidate: Literal["a", "b"] = "a",
    min_win_rate: float = 0.55,
    min_pairs: int = 1,
) -> PreferenceGateResult:
    """Compute preference win-rate gate. Empty corpus → not passed, clear message."""
    parsed: list[PreferencePair] = []
    for p in pairs:
        if isinstance(p, PreferencePair):
            parsed.append(p)
        else:
            parsed.append(PreferencePair.model_validate(p))

    if len(parsed) < min_pairs:
        return PreferenceGateResult(
            passed=False,
            n_pairs=len(parsed),
            min_win_rate=min_win_rate,
            message=(
                "No preference pairs available. Wire DATA-01 annotated corpus "
                "(see configs/quality_mqm.yaml). Gate does not invent labels."
            ),
        )

    wins = sum(1 for p in parsed if p.preferred == candidate)
    ties = sum(1 for p in parsed if p.preferred == "tie")
    decisive = len(parsed) - ties
    win_rate = (wins / decisive) if decisive else 0.0
    passed = decisive > 0 and win_rate >= min_win_rate
    return PreferenceGateResult(
        passed=passed,
        n_pairs=len(parsed),
        win_rate_a=win_rate if candidate == "a" else (1.0 - win_rate if decisive else None),
        min_win_rate=min_win_rate,
        message="passed" if passed else f"win_rate={win_rate:.3f} < min={min_win_rate}",
        details={"wins": wins, "ties": ties, "decisive": decisive, "candidate": candidate},
    )
