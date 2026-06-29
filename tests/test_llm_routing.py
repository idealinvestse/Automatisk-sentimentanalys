"""Tests for LLM model routing."""

from __future__ import annotations

import json

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


def test_select_model_string_tier():
    model = select_model("fast", segment_count=10)
    assert "small" in model or "mistral" in model


def test_select_model_uses_catalog_when_default_present(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    default = "mistralai/mistral-small-3.1-24b-instruct"
    catalog_path.write_text(
        json.dumps({"models": [{"id": default}, {"id": "other/model"}]}),
        encoding="utf-8",
    )
    model = select_model(RoutingTier.FAST, segment_count=10, catalog_path=catalog_path)
    assert model == default


def test_select_model_catalog_fallback_to_any_mistral(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    fallback = "mistralai/mistral-medium-3.5"
    catalog_path.write_text(
        json.dumps({"models": [{"id": fallback}]}),
        encoding="utf-8",
    )
    model = select_model(RoutingTier.FAST, segment_count=10, catalog_path=catalog_path)
    assert model == fallback


def test_select_model_invalid_catalog_falls_back_to_default(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text("not-json", encoding="utf-8")
    model = select_model(RoutingTier.BALANCED, segment_count=10, catalog_path=catalog_path)
    assert "mistral" in model
