from __future__ import annotations

import pytest

from src.core.errors import LLMError
from src.llm.response_validation import validate_call_llm_response


def test_response_validation_requires_requested_task() -> None:
    with pytest.raises(LLMError) as exc_info:
        validate_call_llm_response(
            {"trajectory": {}},
            tasks=["trajectory", "root_cause"],
            segments=[{"text": "Kunden fick en felaktig faktura."}],
        )
    assert exc_info.value.error_code == "llm_task_coverage_failed"


def test_response_validation_accepts_exact_evidence() -> None:
    result = {
        "root_cause": {
            "primary_cause": "Felaktig faktura",
            "evidence_spans": [{"text": "felaktig faktura", "segment_id": 0}],
        }
    }
    assert (
        validate_call_llm_response(
            result,
            tasks=["root_cause"],
            segments=[{"text": "Jag har fått en felaktig faktura igen."}],
        )
        == result
    )


def test_response_validation_rejects_fabricated_evidence() -> None:
    with pytest.raises(LLMError) as exc_info:
        validate_call_llm_response(
            {
                "agent_assessment": {
                    "evidence_spans": [{"text": "Jag lovar en återbetalning", "segment_id": 0}]
                }
            },
            tasks=["agent_assessment"],
            segments=[{"text": "Jag ska undersöka fakturan."}],
        )
    assert exc_info.value.error_code == "llm_evidence_validation_failed"
