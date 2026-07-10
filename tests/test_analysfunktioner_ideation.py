"""Honest degradation, provenance, CCP, aspect platform, partial path, quality OS."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import yaml

from src.analysis.aspect_platform import (
    attach_aspect_platform,
    derive_call_sentiment_from_aspects,
    prefer_aspect_claims,
)
from src.analysis.ccp import evaluate_deep_path_ccps, select_analyzers_runtime
from src.analysis.deep_path import (
    LLM_SUPERSEDED_ANALYZERS,
    inject_unavailable_markers,
    is_unavailable,
    unavailable_payload,
)
from src.analysis.provenance import apply_llm_overrides_with_provenance, build_override_provenance
from src.analysis.registry import resolve_analyzers_for_profile
from src.core.models import Segment
from src.pipeline import CallAnalysisPipeline
from src.quality.mqm import MqmAnnotation, MqmError, PreferencePair, evaluate_preference_gate


def test_callcenter_profile_is_slim_core_plus_sensors():
    selected = resolve_analyzers_for_profile("callcenter")
    assert "sentiment" in selected
    assert "aspect" in selected
    assert "customer_effort" in selected
    for name in LLM_SUPERSEDED_ANALYZERS:
        assert name not in selected


def test_unavailable_payload_shape():
    p = unavailable_payload("root_cause")
    assert p["status"] == "unavailable"
    assert p["value"] is None
    assert is_unavailable(p)


def test_inject_unavailable_when_deep_path_off():
    results = {"sentiment": []}
    inject_unavailable_markers(results, deep_path_active=False, llm_used=False)
    assert is_unavailable(results["empathy"])
    assert is_unavailable(results["root_cause"])
    assert is_unavailable(results["actionable_coaching"])
    assert results["degradation"]["mode"] == "honest"


def test_inject_skips_when_llm_used():
    results = {"empathy": {"overall_empathy": 80}}
    inject_unavailable_markers(results, deep_path_active=True, llm_used=True)
    assert results["empathy"]["overall_empathy"] == 80


def test_prefer_refined_aspects_over_local():
    results = {
        "aspect": [{"aspect": "annat", "sentiment": "neutral", "score": 0.1, "evidence": "x"}],
        "llm": {
            "refined_aspects": [
                {
                    "aspect": "fakturering_pris",
                    "sentiment": "negativ",
                    "score": 0.9,
                    "evidence": [{"text": "fel faktura"}],
                }
            ]
        },
    }
    claims = prefer_aspect_claims(results)
    assert claims[0]["source"] == "llm_refined"
    assert claims[0]["aspect"] == "fakturering_pris"


def test_derive_call_sentiment_from_aspects():
    claims = [
        {"aspect": "a", "sentiment": "negative", "score": 0.8},
        {"aspect": "b", "sentiment": "negative", "score": 0.7},
    ]
    derived = derive_call_sentiment_from_aspects(claims)
    assert derived["label"] == "negative"
    assert derived["aspect_count"] == 2


def test_attach_aspect_platform_writes_fields():
    results = {
        "aspect": [
            {
                "aspect": "agent_attityd",
                "sentiment": "positive",
                "score": 0.8,
                "evidence": "tack så mycket",
            }
        ]
    }
    attach_aspect_platform(results)
    assert results["aspect_claims"]
    assert results["derived_call_sentiment"]["label"] == "positive"


def test_override_provenance_on_llm_merge():
    results = {
        "agent_assessment_local": {"empathy_score": 0.4, "source": "local"},
        "aspect": [{"aspect": "x", "sentiment": "negativ"}],
        "emotion": [{"primary": "frustration"}],
    }
    llm = {
        "meta": {"llm_used": True},
        "agent_assessment": {
            "empathy_score": 0.7,
            "evidence_spans": [{"text": "jag förstår"}],
        },
        "refined_aspects": [{"aspect": "x", "sentiment": "negativ", "evidence": []}],
        "emotion_trajectory": [{"turn": 0, "sentiment": -0.5}],
    }
    apply_llm_overrides_with_provenance(results, llm)
    assert results["override_provenance"]
    fields = {p["field"] for p in results["override_provenance"]}
    assert "agent_assessment" in fields
    assert results["agent_assessment"]["empathy_score"] == 0.7


def test_ccp_fails_on_pii_error():
    segs = [Segment(start=0, end=1, text="hej hej hej hej", speaker="a")] * 4
    ccp = evaluate_deep_path_ccps(segs, {"pii_redaction": {"error": "boom"}})
    assert not ccp.passed
    assert "pii_clean" in ccp.failed_names()


def test_ccp_passes_clean_call():
    segs = [Segment(start=0, end=1, text="detta är ett längre segment", speaker="a")] * 4
    ccp = evaluate_deep_path_ccps(segs, {})
    assert ccp.passed


def test_living_routing_short_call_trims():
    prior = list(resolve_analyzers_for_profile("callcenter")) + ["summary", "topics"]
    runtime = select_analyzers_runtime(prior, segment_count=2)
    assert "summary" not in runtime
    assert "sentiment" in runtime


def test_living_routing_long_call_adds_summary():
    prior = resolve_analyzers_for_profile("callcenter")
    runtime = select_analyzers_runtime(prior, segment_count=12)
    assert "summary" in runtime
    assert "topics" in runtime


def test_partial_analysis_merges_and_marks_incremental():
    pipe = CallAnalysisPipeline(profile="callcenter", device="cpu")
    segs1 = [{"text": "Min faktura är fel", "speaker": "Kund", "start": 0, "end": 1}]
    with (
        patch.object(pipe, "_run_fas4_enrichment", return_value={}),
        patch("src.pipeline.run_registry_analyzers") as mock_run,
    ):
        mock_run.return_value = {
            "sentiment": [{"label": "negative", "score": 0.8}],
            "aspect": [
                {
                    "aspect": "fakturering_pris",
                    "sentiment": "negative",
                    "score": 0.8,
                    "evidence": "faktura",
                    "evidence_spans": [{"text": "faktura"}],
                }
            ],
            "intent": [],
        }
        # Bypass heavy local path internals by stubbing _run_local_analysis
        with patch.object(
            pipe,
            "_run_local_analysis",
            return_value=(
                [Segment(start=0, end=1, text="Min faktura är fel", speaker="Kund")],
                mock_run.return_value,
                None,
            ),
        ):
            r1 = pipe.analyze_segments_partial(segs1, reconcile=False)
            assert r1.results["partial"]["incremental"] is True
            assert r1.results["partial"]["reconciled"] is False
            r2 = pipe.analyze_segments_partial(
                [{"text": "Tack för hjälpen", "speaker": "Kund", "start": 1, "end": 2}],
                previous_results=r1.results,
                reconcile=False,
            )
            assert len(r2.results["sentiment"]) >= 2


def test_reconcile_hook_calls_fas4():
    pipe = CallAnalysisPipeline(profile="callcenter", device="cpu")
    with patch.object(pipe, "_run_fas4_enrichment", return_value={"meta": {"llm_used": False}}) as m:
        out = pipe.reconcile_partial_with_holistic([], {"sentiment": []})
        assert m.called
        assert out["meta"]["llm_used"] is False


def test_mqm_and_preference_gate_empty_corpus():
    ann = MqmAnnotation(
        call_id="c1",
        errors=[MqmError(error_type="aspect_wrong", severity="major", span_text="x")],
    )
    assert ann.errors[0].error_type == "aspect_wrong"
    gate = evaluate_preference_gate([], min_pairs=1)
    assert gate.passed is False
    assert "DATA-01" in gate.message


def test_preference_gate_with_pairs():
    pairs = [
        PreferencePair(call_id="1", output_a_id="a", output_b_id="b", preferred="a"),
        PreferencePair(call_id="2", output_a_id="a", output_b_id="b", preferred="a"),
    ]
    gate = evaluate_preference_gate(pairs, min_win_rate=0.55)
    assert gate.passed is True


def test_analyzer_profiles_yaml_callcenter_optional_has_superseded():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "configs" / "analyzer_profiles.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    optional = set(data["callcenter"]["optional"])
    for name in LLM_SUPERSEDED_ANALYZERS:
        assert name in optional
