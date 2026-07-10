"""Dedicated unit tests for under-tested registry analyzers (core coverage)."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.analysis.aspect import AspectAnalyzer
from src.analysis.empathy_scoring import EmpathyScoringAnalyzer
from src.analysis.resolution_probability import ResolutionProbabilityPredictor
from src.core.models import AnalysisContext, Segment


def _seg(text: str, start: float = 0.0, end: float = 1.0) -> Segment:
    return Segment(text=text, start=start, end=end, speaker="agent")


def test_aspect_empty_segments() -> None:
    assert AspectAnalyzer().analyze(AnalysisContext(segments=[])) == []


def test_aspect_detects_billing_keyword() -> None:
    analyzer = AspectAnalyzer()
    analyzer._sentiment = MagicMock()
    analyzer._sentiment.analyze.return_value = [{"label": "negativ", "score": 0.9}]
    ctx = AnalysisContext(
        segments=[_seg("Min faktura är fel och jag vill ha återbetalning")],
        results={"sentiment": [{"label": "negativ", "score": 0.9}]},
    )
    out = analyzer.analyze(ctx)
    assert out
    assert any(item.get("aspect") == "fakturering_pris" for item in out)


def test_empathy_empty_and_positive_language() -> None:
    empty = EmpathyScoringAnalyzer().analyze(AnalysisContext(segments=[]))
    assert empty["overall_empathy"] == 50

    ctx = AnalysisContext(
        segments=[_seg("Jag förstår, det måste vara frustrerande. Jag ska hjälpa dig.")],
        results={
            "sentiment": [{"label": "positiv", "score": 0.8}],
            "negation": [],
        },
    )
    scored = EmpathyScoringAnalyzer().analyze(ctx)
    assert scored["overall_empathy"] > 50
    assert scored["per_segment"]


def test_resolution_probability_improving_sentiment() -> None:
    ctx = AnalysisContext(
        segments=[_seg("a"), _seg("b"), _seg("c"), _seg("d"), _seg("e")],
        results={
            "sentiment": [
                {"label": "negativ"},
                {"label": "neutral"},
                {"label": "positiv"},
                {"label": "positiv"},
                {"label": "positiv"},
            ],
            "customer_effort": {"overall_ces": 30},
            "empathy": {"overall_empathy": 70},
        },
    )
    out = ResolutionProbabilityPredictor().analyze(ctx)
    assert out["resolution_probability"] > 65
    assert "recommended_action" in out


def test_resolution_probability_empty() -> None:
    out = ResolutionProbabilityPredictor().analyze(AnalysisContext(segments=[]))
    assert out["resolution_probability"] == 50
