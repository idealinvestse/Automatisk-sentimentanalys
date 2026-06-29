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


def test_select_model_from_catalog_when_default_present(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        '{"models": [{"id": "mistralai/mistral-medium-3.5"}]}',
        encoding="utf-8",
    )
    model = select_model(RoutingTier.BALANCED, catalog_path=catalog_path)
    assert model == "mistralai/mistral-medium-3.5"


def test_select_model_falls_back_to_available_mistral(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        '{"models": [{"id": "mistralai/mistral-small-3.1-24b-instruct"}]}',
        encoding="utf-8",
    )
    model = select_model(RoutingTier.DEEP, catalog_path=catalog_path)
    assert "mistral" in model


def test_select_model_uses_data_key_in_catalog(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        '{"data": [{"id": "mistralai/mistral-large-2512"}]}',
        encoding="utf-8",
    )
    model = select_model(RoutingTier.DEEP, catalog_path=catalog_path)
    assert model == "mistralai/mistral-large-2512"


def test_select_model_invalid_catalog_falls_back_to_default(tmp_path):
    catalog_path = tmp_path / "bad.json"
    catalog_path.write_text("not-json", encoding="utf-8")
    model = select_model(RoutingTier.FAST, catalog_path=catalog_path)
    assert model == "mistralai/mistral-small-3.1-24b-instruct"


def test_select_model_string_tier():
    model = select_model("balanced", segment_count=10)
    assert "mistral" in model
