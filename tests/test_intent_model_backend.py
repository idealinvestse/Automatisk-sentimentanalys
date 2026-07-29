"""Tests for optional model-backed intent classification."""

from __future__ import annotations

from pathlib import Path

from src.analysis.resources import ModelResourcePool
from src.intent import IntentClassifier


def test_auto_falls_back_when_model_directory_is_missing(tmp_path: Path) -> None:
    classifier = IntentClassifier(backend="auto", model_path=str(tmp_path / "missing"))

    assert classifier.resolved_backend == "heuristic"
    assert classifier.classify("Jag vill ändra min adress")[0] == "account_update"


def test_model_load_failure_falls_back_to_heuristic(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    classifier = IntentClassifier(backend="model", model_path=str(model_dir))

    assert classifier.resolved_backend == "heuristic"


def test_resource_pool_accepts_auto_and_caches() -> None:
    pool = ModelResourcePool()

    first = pool.get_intent_classifier("auto")
    second = pool.get_intent_classifier("auto")

    assert first is second
    assert first.resolved_backend == "heuristic"
