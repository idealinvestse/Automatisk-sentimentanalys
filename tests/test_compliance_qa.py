"""Tests for Fas 4.2 compliance QA scoring."""

from __future__ import annotations

from src.compliance_qa import QAScorer, load_scorecard

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
