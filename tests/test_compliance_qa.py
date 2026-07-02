"""Tests for Fas 4.2 compliance QA scoring."""

from __future__ import annotations

from unittest.mock import patch

from src.compliance_qa import QAScorer, _score_with_llm_if_needed, load_scorecard
from src.core.models import Segment

SEGMENTS = [
    {"speaker": "SPEAKER_1", "text": "Hej och välkommen till kundtjänst, hur kan jag hjälpa dig?"},
    {"speaker": "SPEAKER_0", "text": "Min faktura är fel, jag är frustrerad."},
    {"speaker": "SPEAKER_1", "text": "Jag förstår och beklagar besväret. Jag kollar direkt."},
]

ROLE_MAP = {"SPEAKER_0": "customer", "SPEAKER_1": "agent"}


class TestComplianceQA:
    def test_load_scorecard(self):
        card = load_scorecard("standard_support_v1")
        assert "criteria" in card
        assert card.get("name") or card.get("scorecard_name")

    def test_score_conversation_rule_based(self):
        scorer = QAScorer(scorecard_path="standard_support_v1")
        result = scorer.score_conversation(SEGMENTS, role_map=ROLE_MAP)
        assert 0 <= result.overall_qa_score <= 100
        assert result.risk_level in ("low", "medium", "high", "critical")
        assert len(result.criteria_results) > 0
        assert result.summary_for_coach

    def test_score_with_local_signals(self):
        scorer = QAScorer(scorecard_path="standard_support_v1")
        signals = {
            "agent_performance": {
                "agent": {"empathy_score": 0.8, "compliance_flags": []},
            }
        }
        result = scorer.score_conversation(SEGMENTS, role_map=ROLE_MAP, local_signals=signals)
        assert isinstance(result.passed_criteria, list)
        assert isinstance(result.failed_criteria, list)

    def test_local_signals_boost_empathy_criterion(self):
        scorer = QAScorer(scorecard_path="standard_support_v1")
        base = scorer.score_conversation(SEGMENTS, role_map=ROLE_MAP)
        boosted = scorer.score_conversation(
            SEGMENTS,
            role_map=ROLE_MAP,
            local_signals={"agent_performance": {"agent": {"empathy_score": 0.95}}},
        )
        empathy_base = next(c for c in base.criteria_results if c.id == "empathy")
        empathy_boost = next(c for c in boosted.criteria_results if c.id == "empathy")
        assert empathy_boost.score >= empathy_base.score

    def test_local_compliance_flags_penalize_criteria(self):
        scorer = QAScorer(scorecard_path="standard_support_v1")
        flagged = scorer.score_conversation(
            SEGMENTS,
            role_map=ROLE_MAP,
            local_signals={
                "agent_performance": {"agent": {"compliance_flags": ["unauthorized_promise"]}},
            },
        )
        tone = next(c for c in flagged.criteria_results if c.id == "tone_professional")
        assert tone.score <= 0.4
        assert tone.passed is False


class TestPiiSafetyInLlmQaPath:
    """Fas A regression tests: PII redaction must not leak unredacted data to LLM."""

    def test_llm_path_aborts_when_redaction_fails(self):
        """If redact_segments raises, the LLM QA path must NOT send original segments.

        Previously, a redaction failure fell back to the original (unredacted)
        segments and forwarded them to the LLM. That violates the GDPR promise
        of 'redact before LLM'. The fix: return the rule-based result instead
        of calling the LLM with unredacted data.
        """
        criterion = {
            "id": "empathy",
            "description": "Agenten visar empati",
            "detection_method": "llm",
            "weight": 10,
            "keywords": ["beklagar", "förstår"],
        }

        class _FailingClient:
            chat_completion_called = False

            def chat_completion(self, **kwargs):
                self.chat_completion_called = True
                return "{}", {}

        class _StubAnalyzer:
            client = _FailingClient()

        segments_with_pii = [
            {"speaker": "SPEAKER_1", "text": "Kund 4111111111111111 ringde, beklagar besväret."},
        ]

        with patch("src.compliance_qa.redact_segments", side_effect=RuntimeError("redaction boom")):
            sc, pas, ev, spans, llm_used = _score_with_llm_if_needed(
                criterion,
                segments_with_pii,
                role_map={"SPEAKER_1": "agent"},
                analyzer=_StubAnalyzer(),
                profile_name="callcenter",
            )

        # The LLM must NOT have been called with unredacted PII.
        assert _StubAnalyzer.client.chat_completion_called is False  # type: ignore[attr-defined]
        # We fall back to rule-based scoring (no LLM).
        assert llm_used is False
        # Score is still produced (rule-based fallback).
        assert 0.0 <= sc <= 1.0

    def test_llm_path_handles_segment_objects_in_transcript_builder(self):
        """The transcript slice builder must not crash on list[Segment] input.

        Previously the generator at compliance_qa.py:270 called ``s.get(...)``
        unconditionally, which crashes on ``Segment`` objects (no ``.get``)
        when redaction fails and falls back to the original segments.
        """
        criterion = {
            "id": "empathy",
            "description": "Agenten visar empati",
            "detection_method": "llm",
            "weight": 10,
            "keywords": ["beklagar"],
        }

        class _RecordingClient:
            received_user: str = ""

            def chat_completion(self, *, messages, **kwargs):
                self.received_user = messages[-1]["content"]
                return '{"score": 0.8, "passed": true, "evidence": ["beklagar"]}', {}

        class _StubAnalyzer:
            client = _RecordingClient()

        segment_objects = [
            Segment(start=0.0, end=1.0, text="Jag beklagar besväret.", speaker="SPEAKER_1"),
        ]

        # Simulate redaction failure -> falls back to original list[Segment].
        # The transcript builder must handle Segment objects without crashing.
        with patch("src.compliance_qa.redact_segments", side_effect=RuntimeError("redaction boom")):
            sc, pas, ev, spans, llm_used = _score_with_llm_if_needed(
                criterion,
                segment_objects,
                role_map={"SPEAKER_1": "agent"},
                analyzer=_StubAnalyzer(),
                profile_name="callcenter",
            )

        # Redaction failed -> must NOT call LLM (PII safety), fall back to rules.
        assert llm_used is False
        assert _StubAnalyzer.client.received_user == ""  # type: ignore[attr-defined]
        # Rule-based fallback still produces a valid score.
        assert 0.0 <= sc <= 1.0
