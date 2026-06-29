"""Tests for LLMJudgeAnalyzer (Task A.3).

All LLM calls are fully mocked — no real network or API keys required.
Covers confidence routing, batching, budget, fallback, provider selection, schema validation, and logging.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.analysis.llm_judge import DEFAULT_MIN_CONFIDENCE, LLMJudgeAnalyzer
from src.core.models import AnalysisContext, Segment
from src.llm.schemas import LLMJudgeResult, LLMJudgeVerdict


def _make_segment(text: str, idx: int = 0) -> Segment:
    return Segment(start=idx * 2.0, end=(idx + 1) * 2.0, text=text)


def _make_sentiment(label: str, score: float) -> dict[str, Any]:
    return {"label": label, "score": score}


def _make_ctx(segments: list[Segment], sentiments: list[dict[str, Any]]) -> AnalysisContext:
    ctx = AnalysisContext(segments=segments)
    ctx.results["sentiment"] = sentiments
    return ctx


@pytest.fixture
def high_conf_ctx() -> AnalysisContext:
    segs = [_make_segment("Tack för hjälpen idag", 0), _make_segment("Det låter bra", 1)]
    sents = [_make_sentiment("positive", 0.92), _make_sentiment("positive", 0.88)]
    return _make_ctx(segs, sents)


@pytest.fixture
def low_conf_ctx() -> AnalysisContext:
    segs = [
        _make_segment("Jag är inte säker på det här", 0),
        _make_segment("Kanske det funkar", 1),
        _make_segment("Det var konstigt", 2),
    ]
    sents = [
        _make_sentiment("neutral", 0.41),
        _make_sentiment("positive", 0.33),
        _make_sentiment("negative", 0.55),
    ]
    return _make_ctx(segs, sents)


@pytest.fixture
def mixed_conf_ctx() -> AnalysisContext:
    segs = [
        _make_segment("Tack", 0),
        _make_segment("Jag förstår inte", 1),
        _make_segment("Kan du upprepa", 2),
        _make_segment("Okej då", 3),
    ]
    sents = [
        _make_sentiment("positive", 0.95),
        _make_sentiment("negative", 0.28),
        _make_sentiment("neutral", 0.51),
        _make_sentiment("positive", 0.89),
    ]
    return _make_ctx(segs, sents)


def test_skips_high_confidence_segments(high_conf_ctx: AnalysisContext) -> None:
    """Segments with confidence ≥ threshold are skipped (no LLM call)."""
    analyzer = LLMJudgeAnalyzer(min_confidence=0.6)
    result = analyzer.analyze(high_conf_ctx)
    assert isinstance(result, LLMJudgeResult)
    assert result.triggered_segments == 0
    assert result.skipped_segments == 2
    assert result.verdicts == []
    assert result.fallback_used is False


def test_judges_low_confidence_segments(low_conf_ctx: AnalysisContext) -> None:
    """Segments below threshold get LLM verdicts via mocked client."""
    analyzer = LLMJudgeAnalyzer(min_confidence=0.6)

    # Inject a mock client that returns deterministic verdicts
    mock_client = MagicMock()
    mock_client.structured_chat.return_value = (
        [
            {
                "segment_index": 0,
                "judge_label": "negative",
                "judge_confidence": 0.71,
                "reasoning": "Negativ markör 'inte säker'",
            },
            {
                "segment_index": 1,
                "judge_label": "neutral",
                "judge_confidence": 0.68,
                "reasoning": "Tveksam formulering",
            },
            {
                "segment_index": 2,
                "judge_label": "negative",
                "judge_confidence": 0.82,
                "reasoning": "Konstig upplevelse",
            },
        ],
        {"cost_usd": 0.0008, "latency_ms": 142},
    )
    analyzer._client = mock_client

    result = analyzer.analyze(low_conf_ctx)
    assert result.triggered_segments == 3
    assert result.skipped_segments == 0
    assert len(result.verdicts) == 3
    assert result.verdicts[0].judge_label == "negative"
    assert result.verdicts[0].original_confidence < 0.6
    assert result.fallback_used is False
    assert result.total_cost_usd > 0


def test_respects_max_segments_per_call(mixed_conf_ctx: AnalysisContext) -> None:
    """Batching respects max_segments_per_call (2 low-conf segments → 2 separate calls for batch size 1)."""
    analyzer = LLMJudgeAnalyzer(min_confidence=0.6, max_segments_per_call=1)

    mock_client = MagicMock()
    # Two batches → two structured_chat calls
    mock_client.structured_chat.side_effect = [
        (
            [
                {
                    "segment_index": 0,
                    "judge_label": "negative",
                    "judge_confidence": 0.75,
                    "reasoning": "r1",
                }
            ],
            {"cost_usd": 0.0004},
        ),
        (
            [
                {
                    "segment_index": 0,
                    "judge_label": "neutral",
                    "judge_confidence": 0.66,
                    "reasoning": "r2",
                }
            ],
            {"cost_usd": 0.0004},
        ),
    ]
    analyzer._client = mock_client

    result = analyzer.analyze(mixed_conf_ctx)
    # 2 low-conf segments → 2 calls when batch=1
    assert mock_client.structured_chat.call_count == 2
    assert result.triggered_segments == 2
    assert len(result.verdicts) == 2


def test_budget_exceeded_stops_llm(low_conf_ctx: AnalysisContext) -> None:
    """When estimated budget would be exceeded, stop early and set budget_exceeded=True."""
    # The _estimate_cost(800,120) ≈ 0.000094. Use a budget smaller than that so guard triggers on first batch.
    analyzer = LLMJudgeAnalyzer(min_confidence=0.6, max_cost_usd=0.00001)

    mock_client = MagicMock()
    mock_client.structured_chat.return_value = ([], {"cost_usd": 0.05})
    analyzer._client = mock_client

    result = analyzer.analyze(low_conf_ctx)
    assert result.budget_exceeded is True
    assert result.fallback_used is False
    assert result.verdicts == []


def test_llm_failure_falls_back_gracefully(low_conf_ctx: AnalysisContext) -> None:
    """Any exception from LLM client → fallback_used=True + mock verdicts (graceful, no crash)."""
    analyzer = LLMJudgeAnalyzer(min_confidence=0.6)

    mock_client = MagicMock()
    mock_client.structured_chat.side_effect = RuntimeError("rate limit")
    analyzer._client = mock_client

    result = analyzer.analyze(low_conf_ctx)
    assert result.fallback_used is True
    assert result.triggered_segments == 3
    # Internal _mock_judge_response still produces usable verdicts (resilience design)
    assert len(result.verdicts) == 3
    assert all(v.model == "llama-3.1-8b-instant" for v in result.verdicts)


def test_provider_routing() -> None:
    """Provider flag selects correct client class (groq vs openrouter)."""
    # Groq path
    with patch("src.analysis.llm_judge.GroqClient") as mock_groq:
        mock_groq.return_value = MagicMock()
        a_groq = LLMJudgeAnalyzer(provider="groq")
        a_groq._get_client()
        mock_groq.assert_called_once()

    # OpenRouter path (default)
    with patch("src.analysis.llm_judge.OpenRouterClient") as mock_or:
        mock_or.return_value = MagicMock()
        a_or = LLMJudgeAnalyzer(provider="openrouter")
        a_or._get_client()
        mock_or.assert_called_once()


def test_default_threshold_is_0_6() -> None:
    """Verify the documented default threshold."""
    analyzer = LLMJudgeAnalyzer()
    assert analyzer.min_confidence == DEFAULT_MIN_CONFIDENCE
    assert analyzer.min_confidence == 0.6


def test_verdict_schema_validates() -> None:
    """Pydantic schema accepts valid input and rejects invalid."""
    valid = LLMJudgeVerdict(
        segment_index=0,
        original_sentiment="negative",
        original_confidence=0.33,
        judge_label="negative",
        judge_confidence=0.78,
        reasoning="Negativ ton.",
        model="llama-3.1-8b-instant",
    )
    assert valid.judge_label == "negative"

    # Missing required field
    with pytest.raises(ValidationError):
        LLMJudgeVerdict(
            segment_index=0,
            original_sentiment="positive",
            original_confidence=0.9,
            judge_label="positive",
            judge_confidence=0.9,
            reasoning="ok",
            # model missing
        )


def test_external_llm_call_logged(
    low_conf_ctx: AnalysisContext, caplog: pytest.LogCaptureFixture
) -> None:
    """Every LLM egress produces an 'EXTERNAL LLM CALL' log line."""
    analyzer = LLMJudgeAnalyzer(min_confidence=0.6)
    mock_client = MagicMock()
    mock_client.structured_chat.return_value = ([], {"cost_usd": 0.0})
    analyzer._client = mock_client

    with caplog.at_level(logging.INFO):
        analyzer.analyze(low_conf_ctx)

    assert any("EXTERNAL LLM CALL (LLMJudge)" in rec.message for rec in caplog.records)
