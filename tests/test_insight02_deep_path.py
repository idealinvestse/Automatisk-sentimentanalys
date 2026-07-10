"""INSIGHT-02: verify LLM deep path skips superseded local analyzers."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.analysis.deep_path import LLM_SUPERSEDED_ANALYZERS, filter_superseded
from src.core.models import Segment
from src.pipeline_steps import PipelineLLMContext, should_use_any_llm


def test_llm_superseded_set_matches_analyzer_strategy():
    """Must stay aligned with docs/ANALYZER_STRATEGY.md tier table."""
    assert LLM_SUPERSEDED_ANALYZERS == frozenset(
        {
            "empathy",
            "trajectory",
            "insights",
            "root_cause",
            "actionable_coaching",
        }
    )


def test_filter_superseded_skips_all_llm_tier_analyzers():
    selected = sorted(LLM_SUPERSEDED_ANALYZERS) + ["sentiment", "intent"]
    filtered = filter_superseded(selected, skip=True)
    assert not set(filtered or []) & LLM_SUPERSEDED_ANALYZERS
    assert "sentiment" in filtered
    assert "intent" in filtered


def test_should_use_any_llm_for_callcenter_long_call():
    ctx = PipelineLLMContext(
        profile="callcenter",
        provider="openrouter",
        use_mistral_llm=False,
        deep_analysis=False,
        llm_model=None,
        llm_api_key=None,
        groq_eu_residency=False,
    )
    segments = [
        Segment(start=float(i), end=float(i + 1), text=f"seg {i}", speaker="agent")
        for i in range(8)
    ]
    assert should_use_any_llm(segments, ctx) is True


def test_pipeline_passes_skip_llm_superseded_when_llm_active():
    """CallAnalysisPipeline must skip superseded analyzers when LLM path is active."""
    from src.pipeline import CallAnalysisPipeline

    pipe = CallAnalysisPipeline(
        profile="callcenter",
        provider="openrouter",
        use_mistral_llm=False,
    )
    segments = [{"text": "Hej", "speaker": "agent", "start": 0.0, "end": 1.0}] * 8
    captured: dict = {}

    def _capture_run(*args, **kwargs):
        captured["skip_llm_superseded"] = kwargs.get("skip_llm_superseded")
        return {"sentiment": []}

    with (
        patch.object(pipe, "_build_analyzer_configs", return_value={}),
        patch("src.pipeline.run_registry_analyzers", side_effect=_capture_run),
        patch.object(pipe, "_run_fas4_enrichment", return_value={}),
    ):
        pipe.analyze_segments(segments)

    assert captured.get("skip_llm_superseded") is True


@pytest.mark.parametrize("name", sorted(LLM_SUPERSEDED_ANALYZERS))
def test_each_superseded_analyzer_filtered_individually(name: str):
    assert name not in (filter_superseded([name, "sentiment"], skip=True) or [])
