"""Tests for shared EvidenceSpan contract (#7)."""

from __future__ import annotations

from src.analysis.aspect import AspectAnalyzer
from src.analysis.compliance_risk import ComplianceRiskAnalyzer
from src.analysis.evidence import make_evidence_span
from src.core.models import AnalysisContext, Segment
from src.llm.schemas import CallLLMOutput, EvidenceSpan, OverrideProvenance


def test_make_evidence_span_includes_timing_fields():
    span = make_evidence_span(
        "felaktig faktura",
        speaker_role="customer",
        turn_index=2,
        segment_id=2,
        start=1.5,
        end=3.0,
    )
    assert isinstance(span, EvidenceSpan)
    assert span.text == "felaktig faktura"
    assert span.segment_id == 2
    assert span.start == 1.5
    dumped = span.model_dump()
    assert "segment_id" in dumped
    assert dumped["speaker_role"] == "customer"


def test_aspect_analyzer_emits_evidence_spans():
    segs = [
        Segment(start=0.0, end=2.0, text="Min faktura är felaktig och dyr", speaker="Kund"),
    ]
    ctx = AnalysisContext(
        segments=segs,
        results={"sentiment": [{"label": "negative", "score": 0.9}]},
    )
    out = AspectAnalyzer().analyze(ctx)
    assert out
    assert "evidence" in out[0]
    assert out[0]["evidence_spans"]
    assert out[0]["evidence_spans"][0]["text"]
    assert out[0]["evidence_spans"][0]["segment_id"] == 0


def test_compliance_emits_evidence_spans():
    segs = [
        Segment(start=0.0, end=1.0, text="Jag lovar att det är löst imorgon", speaker="A"),
    ]
    ctx = AnalysisContext(
        segments=segs,
        results={"role": {"roles": {"A": "agent"}}},
    )
    out = ComplianceRiskAnalyzer().analyze(ctx)
    assert out["flagged_segments"]
    assert out["flagged_segments"][0]["evidence_spans"][0]["text"]


def test_llm_schema_accepts_extended_evidence_and_provenance():
    out = CallLLMOutput.model_validate(
        {
            "refined_aspects": [
                {
                    "aspect": "fakturering_pris",
                    "sentiment": "negativ",
                    "score": 0.8,
                    "evidence": [
                        {
                            "text": "fel faktura",
                            "speaker_role": "customer",
                            "segment_id": 1,
                            "start": 0.0,
                            "end": 1.0,
                        }
                    ],
                }
            ],
            "override_provenance": [
                {
                    "field": "refined_aspects",
                    "local_source": "aspect",
                    "reason": "deep_path_holistic",
                    "evidence_spans": [{"text": "fel faktura"}],
                    "channel_diversity_ok": True,
                }
            ],
        }
    )
    assert out.refined_aspects[0].evidence[0].segment_id == 1
    assert isinstance(out.override_provenance[0], OverrideProvenance)
