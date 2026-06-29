"""Unit tests for heuristic registry analyzers (A2)."""

from __future__ import annotations

import pytest

from src.analysis.actionable_coaching import ActionableCoachingAnalyzer
from src.analysis.emotion import EmotionAnalyzer
from src.analysis.multi_turn_journey import MultiTurnJourneyMapper
from src.analysis.negation import NegationAnalyzer
from src.analysis.registry import ensure_analyzers_loaded
from src.analysis.role_classifier import RoleAnalyzer
from src.analysis.schemas import validate_analyzer_result
from src.core.models import AnalysisContext, Segment


def _seg(text: str, speaker: str = "SPEAKER_0", start: float = 0.0) -> Segment:
    return Segment(start=start, end=start + 1.0, text=text, speaker=speaker)


@pytest.fixture(autouse=True)
def _load_analyzers() -> None:
    ensure_analyzers_loaded()


class TestEmotionAnalyzer:
    def test_detects_frustration_keyword(self) -> None:
        ctx = AnalysisContext(
            segments=[_seg("Jag är jättearg på er service!")],
            results={
                "sentiment": [{"label": "negativ", "score": 0.8}],
                "negation": [{"has_negation": False, "negation_count": 0}],
            },
        )
        out = EmotionAnalyzer().analyze(ctx)
        assert out[0]["primary"] in ("frustration", "ilska")

    def test_neutral_when_no_markers(self) -> None:
        ctx = AnalysisContext(
            segments=[_seg("Hej där.")],
            results={
                "sentiment": [{"label": "neutral", "score": 0.0}],
                "negation": [{"has_negation": False, "negation_count": 0}],
            },
        )
        out = EmotionAnalyzer().analyze(ctx)
        assert out[0]["primary"] == "neutral"


class TestNegationAnalyzer:
    def test_detects_swedish_negation(self) -> None:
        ctx = AnalysisContext(segments=[_seg("Det fungerar inte alls.")])
        out = NegationAnalyzer().analyze(ctx)
        assert out[0]["has_negation"] is True
        assert out[0]["negation_count"] >= 1

    def test_list_validation_per_item(self) -> None:
        raw = [
            {"has_negation": True, "negation_count": 1},
            {"has_negation": False, "negation_count": 0},
        ]
        validated = validate_analyzer_result("negation", raw, mode="strict")
        assert isinstance(validated, list)
        assert len(validated) == 2
        assert validated[0]["has_negation"] is True


class TestMultiTurnJourneyMapper:
    def test_uses_intent_and_sentiment_for_escalation(self) -> None:
        segments = [
            _seg("Hej, jag behöver hjälp.", "SPEAKER_1", 0),
            _seg("Min faktura är fel.", "SPEAKER_0", 1),
            _seg("Jag förstår, vi kollar.", "SPEAKER_1", 2),
            _seg("Fortfarande inte hjälpt.", "SPEAKER_0", 3),
        ]
        ctx = AnalysisContext(
            segments=segments,
            results={
                "intent": [
                    {"intent": "support"},
                    {"intent": "complaint"},
                    {"intent": "support"},
                    {"intent": "complaint"},
                ],
                "sentiment": [
                    {"label": "neutral"},
                    {"label": "negativ"},
                    {"label": "neutral"},
                    {"label": "negativ"},
                ],
            },
        )
        out = MultiTurnJourneyMapper().analyze(ctx)
        assert len(out["journey_stages"]) == 4
        assert out["unresolved_count"] >= 1
        assert any(s["stage"] == "escalation" for s in out["journey_stages"])

    def test_too_short_returns_message(self) -> None:
        ctx = AnalysisContext(segments=[_seg("Hej"), _seg("Hej då")])
        out = MultiTurnJourneyMapper().analyze(ctx)
        assert out["journey_stages"] == []


class TestRoleAnalyzer:
    def test_requires_sentiment_dependency(self) -> None:
        assert "sentiment" in RoleAnalyzer().requires

    def test_two_speaker_role_map(self) -> None:
        segments = [
            _seg("Hej och välkommen.", "SPEAKER_1"),
            _seg("Jag har ett problem.", "SPEAKER_0"),
        ]
        ctx = AnalysisContext(
            segments=segments,
            results={"sentiment": [{"label": "neutral"}, {"label": "negativ"}]},
        )
        out = RoleAnalyzer().analyze(ctx)
        assert out["roles"]["SPEAKER_1"] == "agent"
        assert out["roles"]["SPEAKER_0"] == "customer"


class TestActionableCoachingAnalyzer:
    def test_requires_compliance_risk(self) -> None:
        assert "compliance_risk" in ActionableCoachingAnalyzer().requires

    def test_flags_low_empathy(self) -> None:
        ctx = AnalysisContext(
            segments=[_seg("Test")],
            results={
                "empathy": {"overall_empathy": 30},
                "customer_effort": {"overall_ces": 10},
                "trajectory": {"trend": "stable"},
                "compliance_risk": {"overall_risk_level": "low"},
            },
        )
        out = ActionableCoachingAnalyzer().analyze(ctx)
        ids = [i["rule_id"] for i in out["coaching_insights"]]
        assert "low_empathy" in ids
