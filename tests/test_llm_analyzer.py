"""Tests for ConversationMistralAnalyzer (Task 3.1.2) and Pydantic schemas (3.1.3).

Focus:
- Schema validation roundtrip
- Building role-labeled transcript
- Successful path (mocked client) returns validated + meta
- Fallback on client error (never crashes caller)
- Task subsetting
- Evidence and Swedish text handling (basic)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.llm.mistral_analyzer import (
    ConversationMistralAnalyzer,
    _build_role_labeled_transcript,
)
from src.llm.schemas import LLM_OUTPUT_JSON_SCHEMA, CallLLMOutput


def test_schemas_produce_strict_compatible_json_schema():
    """The top-level schema must be usable directly with OpenRouter strict json_schema."""
    schema = LLM_OUTPUT_JSON_SCHEMA
    assert schema["type"] == "object"
    assert "trajectory" in schema["properties"]
    assert "actionable_summary" in schema["properties"]
    # Pydantic v2 with extra=forbid gives additionalProperties in generated schema in many cases
    # We mainly care that it is a valid JSON Schema object the client can send.
    assert "properties" in schema


def test_schema_validation_roundtrip():
    good = {
        "trajectory": {
            "points": [{"turn": 0, "sentiment": -0.4, "primary_emotion": "frustration"}],
            "customer_sentiment_slope": -0.12,
            "escalation_events": ["Kunden upprepade 'jag vill ha pengarna tillbaka'"],
            "summary": "Kundens frustration eskalerade efter tur 3 p.g.a. faktura.",
        },
        "refined_aspects": [
            {
                "aspect": "fakturering_pris",
                "sentiment": "negativ",
                "score": 0.92,
                "evidence": [{"text": "Fakturan var helt fel", "speaker_role": "customer"}],
            }
        ],
        "actionable_summary": {
            "problem": "Kunden fick felaktig faktura och ingen hjälp att korrigera.",
            "recommendations_for_qa": ["Använd 'jag förstår att det är frustrerande' tidigt."],
            "final_customer_state": "Mycket missnöjd",
            "risk_level": "high",
        },
        "agent_assessment": {
            "empathy_score": 0.35,
            "compliance_flags": ["missade att bekräfta kundens känsla"],
            "strengths": ["Svarade snabbt"],
            "weaknesses": ["Låg empati vid fakturaklagomål"],
            "evidence_spans": [
                {"text": "Fakturan var helt fel", "speaker_role": "customer", "turn_index": 0}
            ],
            "specific_coaching_recommendations": [
                {
                    "recommendation": "Säg 'Jag förstår att det är frustrerande' direkt efter kundens klagomål.",
                    "evidence_spans": [
                        {"text": "Fakturan var helt fel", "speaker_role": "customer"}
                    ],
                    "priority": "high",
                    "category": "empathy",
                }
            ],
            "overall_assessment": "Agenten behöver träna empati-fraser vid fakturaärenden.",
        },
        "meta": {"model": "mistralai/mistral-medium-3.5"},
    }

    model = CallLLMOutput.model_validate(good)
    assert model.actionable_summary is not None
    assert model.actionable_summary.risk_level == "high"
    dumped = model.model_dump()
    assert dumped["meta"]["model"] == "mistralai/mistral-medium-3.5"
    # Fas 4.1.2: detailed coaching recs + evidence
    assess = model.agent_assessment
    assert assess is not None
    assert assess.empathy_score == 0.35
    assert len(assess.specific_coaching_recommendations) >= 1
    rec = assess.specific_coaching_recommendations[0]
    assert "recommendation" in rec
    assert "evidence_spans" in rec
    assert assess.overall_assessment is not None


def test_schema_rejects_extra_fields():
    bad = {"trajectory": None, "foo": "bar"}  # extra not allowed
    with pytest.raises(ValidationError):
        CallLLMOutput.model_validate(bad)


def test_build_role_labeled_transcript_basic():
    segments = [
        {"speaker": "SPEAKER_0", "text": "Hej, vad gäller det?"},
        {"speaker": "SPEAKER_1", "text": "Jag har en fråga om fakturan."},
    ]
    role_map = {"SPEAKER_0": "agent", "SPEAKER_1": "customer"}

    txt = _build_role_labeled_transcript(segments, role_map)
    assert "[AGENT] Hej, vad gäller det?" in txt
    assert "[CUSTOMER] Jag har en fråga om fakturan." in txt


def test_analyzer_returns_fallback_on_client_error():
    analyzer = ConversationMistralAnalyzer(client=MagicMock())

    # Make the client blow up
    with patch.object(analyzer.client, "structured_chat", side_effect=RuntimeError("boom")):
        out = analyzer.analyze_full_conversation(
            segments=[{"text": "test", "speaker": "SPEAKER_0"}],
            role_map={"SPEAKER_0": "agent"},
        )

    assert out.get("fallback") is True
    assert "error" in out
    assert out["meta"]["fallback_reason"] == "llm_error"


def test_analyzer_success_path_validates_and_merges_meta():
    fake_client = MagicMock()
    fake_result = {
        "trajectory": {
            "points": [],
            "customer_sentiment_slope": 0.0,
            "escalation_events": [],
            "summary": "Samtalet var lugnt.",
        },
        "actionable_summary": {
            "problem": "Inget större problem.",
            "final_customer_state": "Nöjd",
            "recommendations_for_qa": [],
        },
        "meta": {"from_llm": True},
    }
    fake_meta = {
        "model": "mistralai/mistral-medium-3.5",
        "cost_usd": 0.012,
        "cached": False,
    }
    fake_client.structured_chat.return_value = (fake_result, fake_meta)

    analyzer = ConversationMistralAnalyzer(client=fake_client, model="mistralai/mistral-medium-3.5")

    segments = [
        {"speaker": "SPEAKER_0", "text": "Tack för hjälpen."},
        {"speaker": "SPEAKER_1", "text": "Varsågod."},
    ]
    out = analyzer.analyze_full_conversation(
        segments=segments,
        role_map={"SPEAKER_0": "agent", "SPEAKER_1": "customer"},
        tasks=["trajectory", "actionable_summary"],
    )

    assert out["actionable_summary"]["problem"] == "Inget större problem."
    assert out["meta"]["cost_usd"] == 0.012
    assert out["meta"]["cached"] is False
    assert out["meta"]["model"] == "mistralai/mistral-medium-3.5"
    assert "tasks" in out["meta"]

    # The client must have been called with a json_schema
    call_kwargs = fake_client.structured_chat.call_args.kwargs
    assert "json_schema" in call_kwargs
    assert call_kwargs["json_schema"]["type"] == "object"


# ---------------------------------------------------------------------------
# Prompt quality checks (Task 3.2.1)
# ---------------------------------------------------------------------------

from src.llm import LLM_SYSTEM_PROMPT, build_user_prompt, get_system_prompt


def test_prompts_contain_key_callcenter_and_evidence_instructions():
    sys_p = get_system_prompt()
    assert (
        "EVENSSPAN" in sys_p.upper() or "EVIDENSSPAN" in sys_p.upper() or "evidens" in sys_p.lower()
    )
    assert "KUNDEN" in sys_p.upper() or "customer" in sys_p.lower()
    assert "JSON" in sys_p.upper()

    user_p = build_user_prompt(
        transcript="[AGENT] Hej\n[CUSTOMER] Jag är arg på fakturan",
        tasks=["root_cause", "agent_assessment"],
    )
    assert "root_cause" in user_p or "Root cause" in user_p
    assert "evidens" in user_p.lower() or "citat" in user_p.lower()
    assert "AGENT" in user_p and "CUSTOMER" in user_p


def test_system_prompt_is_stable_and_strong():
    p = LLM_SYSTEM_PROMPT
    assert len(p) > 400  # substantial guidance
    assert "callcenter" in p.lower() or "kundtjänst" in p.lower()


# ---------------------------------------------------------------------------
# Groq analyzer tests (groq_analyzer.py)
# ---------------------------------------------------------------------------

from src.llm.groq_analyzer import GroqAnalyzer
from src.llm.schemas import GROQ_DEFAULT_MODEL, GROQ_MODELS


def test_groq_model_registry_is_complete():
    """GROQ_MODELS contains all 17 expected models with required fields."""
    assert len(GROQ_MODELS) >= 17
    for _model_id, info in GROQ_MODELS.items():
        assert "context_window" in info
        assert "owner" in info
        assert "pricing_in" in info
        assert "pricing_out" in info
        assert "capabilities" in info
        assert "tier" in info

    # Key models must be present
    assert "llama-3.1-8b-instant" in GROQ_MODELS
    assert "llama-3.3-70b-versatile" in GROQ_MODELS
    assert "openai/gpt-oss-20b" in GROQ_MODELS
    assert "openai/gpt-oss-120b" in GROQ_MODELS
    assert GROQ_DEFAULT_MODEL == "llama-3.3-70b-versatile"

    # Pricing sanity check
    llama8b = GROQ_MODELS["llama-3.1-8b-instant"]
    assert llama8b["pricing_in"] < llama8b["pricing_out"]  # output > input
    assert llama8b["pricing_in"] <= 3.0  # not crazy expensive


def test_groq_analyzer_fallback_on_client_error():
    """GroqAnalyzer returns fallback dict when GroqClient errors."""
    analyzer = GroqAnalyzer(client=MagicMock())

    with patch.object(analyzer.client, "structured_chat", side_effect=RuntimeError("boom")):
        out = analyzer.analyze_full_conversation(
            segments=[{"text": "test", "speaker": "SPEAKER_0"}],
            role_map={"SPEAKER_0": "agent"},
            anonymize_before_llm=True,
        )

    assert out.get("fallback") is True
    assert "error" in out
    assert out["meta"]["fallback_reason"] == "groq_llm_error"
    assert out["meta"]["provider"] == "groq"


def test_groq_analyzer_success_path_validates_and_merges_meta():
    """Successful Groq analyzer path returns validated data + meta."""
    fake_client = MagicMock()
    fake_result = {
        "trajectory": {
            "points": [],
            "customer_sentiment_slope": 0.0,
            "escalation_events": [],
            "summary": "Samtalet var lugnt.",
        },
        "actionable_summary": {
            "problem": "Inget större problem.",
            "final_customer_state": "Nöjd",
            "recommendations_for_qa": [],
        },
        "meta": {"from_groq": True},
    }
    fake_meta = {
        "model": "llama-3.3-70b-versatile",
        "cost_usd": 0.0012,
        "cached": False,
        "provider": "groq",
    }
    fake_client.structured_chat.return_value = (fake_result, fake_meta)

    analyzer = GroqAnalyzer(client=fake_client, model="llama-3.3-70b-versatile")

    segments = [
        {"speaker": "SPEAKER_0", "text": "Tack för hjälpen."},
        {"speaker": "SPEAKER_1", "text": "Varsågod."},
    ]
    out = analyzer.analyze_full_conversation(
        segments=segments,
        role_map={"SPEAKER_0": "agent", "SPEAKER_1": "customer"},
        tasks=["trajectory", "actionable_summary"],
        anonymize_before_llm=True,
    )

    assert out["actionable_summary"]["problem"] == "Inget större problem."
    assert out["meta"]["cost_usd"] == 0.0012
    assert out["meta"]["cached"] is False
    assert out["meta"]["provider"] == "groq"
    assert "tasks" in out["meta"]

    call_kwargs = fake_client.structured_chat.call_args.kwargs
    assert "json_schema" in call_kwargs
    assert call_kwargs["anonymize_before_llm"] is True


def test_groq_analyzer_gdpr_gate_blocks_without_anonymize():
    """Without anonymize_before_llm, Groq client should block (GDPR gate)."""
    from src.llm.groq_client import GroqClient as RealGroqClient

    # Use a real client with groq_eu_residency=False
    client = RealGroqClient(api_key="dummy", groq_eu_residency=False)
    analyzer = GroqAnalyzer(client=client, groq_eu_residency=False)

    segments = [
        {"speaker": "SPEAKER_0", "text": "Hej, mitt personnummer är 19900101-1234"},
    ]

    out = analyzer.analyze_full_conversation(
        segments=segments,
        role_map={"SPEAKER_0": "customer"},
        anonymize_before_llm=False,
    )

    # Should be a fallback due to GDPR gate
    assert out.get("fallback") is True
    assert "groq_llm_error" in out.get("meta", {}).get("fallback_reason", "")


def test_groq_analyzer_allows_with_eu_residency():
    """With groq_eu_residency=True, GDPR gate passes."""
    fake_client = MagicMock()
    fake_result = {
        "trajectory": {
            "points": [],
            "customer_sentiment_slope": 0.0,
            "escalation_events": [],
            "summary": "OK",
        },
        "meta": {},
    }
    fake_meta = {
        "model": "llama-3.3-70b-versatile",
        "cost_usd": 0.001,
        "cached": False,
        "provider": "groq",
    }
    fake_client.structured_chat.return_value = (fake_result, fake_meta)

    analyzer = GroqAnalyzer(client=fake_client, groq_eu_residency=True)

    out = analyzer.analyze_full_conversation(
        segments=[{"text": "test", "speaker": "SPEAKER_0"}],
        role_map={"SPEAKER_0": "agent"},
        anonymize_before_llm=False,  # GDPR gate passes because eu_residency=True
    )

    assert out.get("fallback") is not True
    assert out["meta"]["provider"] == "groq"
