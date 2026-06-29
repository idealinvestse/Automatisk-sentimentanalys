"""CI threshold tests for analyzer benchmarks (DATA-01 / analyzer accuracy program)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.benchmark_intent import benchmark_backend, load_intent_jsonl
from scripts.validate_intent_corpus import validate_corpus

CONFIG_PATH = Path("configs/analyzer_eval.yaml")


@pytest.fixture(scope="module")
def eval_config() -> dict:
    if not CONFIG_PATH.is_file():
        pytest.skip("analyzer_eval.yaml missing")
    with CONFIG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


class TestIntentCorpus:
    def test_train_corpus_valid(self) -> None:
        stats = validate_corpus("data/intent_train.jsonl", min_rows=200, min_per_intent=20)
        assert stats["duplicate_ratio"] == 0.0

    def test_val_corpus_valid(self) -> None:
        stats = validate_corpus(
            "data/intent_val.jsonl", min_rows=50, min_per_intent=5, max_class_ratio=0.15
        )
        assert stats["duplicate_ratio"] == 0.0


class TestIntentBenchmarkThresholds:
    def test_heuristic_macro_f1_on_val(self, eval_config: dict) -> None:
        val_file = Path(eval_config["intent"]["val_file"])
        if not val_file.is_file():
            pytest.skip("intent val file missing")
        texts, labels = load_intent_jsonl(val_file)
        metrics = benchmark_backend(texts, labels, backend="heuristic")
        min_f1 = eval_config["intent"]["min_macro_f1"]
        min_acc = eval_config["intent"]["min_accuracy"]
        assert metrics["f1_macro"] >= min_f1, f"macro F1 {metrics['f1_macro']} < {min_f1}"
        assert metrics["accuracy"] >= min_acc, f"accuracy {metrics['accuracy']} < {min_acc}"


class TestBaselineFiles:
    def test_intent_baseline_structure(self) -> None:
        path = Path("reports/intent_baseline.json")
        if not path.is_file():
            pytest.skip("intent baseline not generated yet")
        data = json.loads(path.read_text(encoding="utf-8"))
        heur = data.get("backends", {}).get("heuristic", {})
        assert "f1_macro" in heur
        assert "per_class" in heur
        assert "confusion_matrix" in heur

    def test_analyzer_eval_config_loads(self, eval_config: dict) -> None:
        assert "sentiment" in eval_config
        assert "intent" in eval_config
        assert eval_config["intent"]["min_macro_f1"] >= 0.7
