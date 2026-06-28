"""Tests for LLM model routing."""

from __future__ import annotations

from src.llm.routing import RoutingTier, select_model


def test_select_model_fast_for_short_calls():
    model = select_model(RoutingTier.BALANCED, segment_count=3)
    assert "small" in model or "mistral" in model


def test_select_model_deep_for_long_calls():
    model = select_model(RoutingTier.BALANCED, segment_count=25, deep_analysis=True)
    assert "large" in model or "medium" in model


def test_select_model_budget_forces_fast():
    model = select_model(RoutingTier.DEEP, segment_count=25, budget_usd=0.01)
    assert "small" in model
