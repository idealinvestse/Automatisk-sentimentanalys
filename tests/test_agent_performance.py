"""Tests for Fas 4.1 agent performance engine."""

from __future__ import annotations

import pytest

from src.agent_performance import (
    aggregate_agent_performance,
    aggregate_team_performance,
    compute_call_agent_performance,
    compute_compliance_flags,
    compute_empathy_and_deescalation,
    compute_talk_ratios,
)

SEGMENTS_GOOD = [
    {"start": 0, "end": 3, "text": "Hej och välkommen till kundtjänst.", "speaker": "SPEAKER_1"},
    {"start": 3, "end": 7, "text": "Jag är arg på fakturan!", "speaker": "SPEAKER_0"},
    {
        "start": 7,
        "end": 12,
        "text": "Jag beklagar verkligen. Jag förstår att det är frustrerande. Jag fixar det.",
        "speaker": "SPEAKER_1",
    },
    {"start": 12, "end": 15, "text": "Tack, det var bra.", "speaker": "SPEAKER_0"},
]

SEGMENTS_BAD = [
    {"start": 0, "end": 5, "text": "Jag är extremt arg!", "speaker": "SPEAKER_0"},
    {"start": 5, "end": 15, "text": "Okej. Okej. Okej. Det är så här.", "speaker": "SPEAKER_1"},
]

ROLE_MAP = {"SPEAKER_0": "customer", "SPEAKER_1": "agent"}


class TestAgentPerformanceMetrics:
    def test_talk_ratios(self):
        ratios = compute_talk_ratios(SEGMENTS_GOOD, ROLE_MAP)
        assert 0.0 <= ratios["agent_talk_ratio"] <= 1.0
        assert ratios["customer_talk_ratio"] + ratios["agent_talk_ratio"] == pytest.approx(
            1.0, abs=0.01
        )

    def test_empathy_good_agent(self):
        scores = compute_empathy_and_deescalation(SEGMENTS_GOOD, ROLE_MAP)
        assert scores["empathy_score"] >= 0.4
        assert scores["de_escalation_effectiveness"] >= 0.0

    def test_compliance_flags_bad_agent(self):
        flags = compute_compliance_flags(SEGMENTS_BAD, ROLE_MAP)
        assert isinstance(flags, list)
        assert len(flags) >= 1

    def test_compute_call_agent_performance(self):
        perf = compute_call_agent_performance(SEGMENTS_GOOD, role_map=ROLE_MAP)
        assert perf.agent.empathy_score >= 0.4
        assert perf.customer.talk_ratio >= 0.0
        assert isinstance(perf.local_coaching_hints, list)

    def test_cache_returns_same_result(self):
        p1 = compute_call_agent_performance(SEGMENTS_GOOD, role_map=ROLE_MAP)
        p2 = compute_call_agent_performance(SEGMENTS_GOOD, role_map=ROLE_MAP)
        assert p1.agent.empathy_score == p2.agent.empathy_score

    def test_aggregate_agent_performance(self):
        perfs = [
            compute_call_agent_performance(SEGMENTS_GOOD, role_map=ROLE_MAP),
            compute_call_agent_performance(SEGMENTS_BAD, role_map=ROLE_MAP),
        ]
        agg = aggregate_agent_performance(perfs, agent_id="Agent-1")
        assert agg["agent_id"] == "Agent-1"
        assert agg["call_count"] == 2
        assert "averages" in agg

    def test_aggregate_team_performance(self):
        perfs = [compute_call_agent_performance(SEGMENTS_GOOD, role_map=ROLE_MAP)]
        agg = aggregate_team_performance(perfs)
        assert agg["team_call_count"] == 1
        assert "team_averages" in agg
