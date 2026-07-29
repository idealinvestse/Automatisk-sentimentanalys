"""Tests for intent training configuration and provenance helpers."""

from __future__ import annotations

from pathlib import Path

from scripts.train_intent import corpus_sha256, load_config


def test_load_intent_training_config() -> None:
    config = load_config("configs/intent_finetune.yaml")

    assert config["base_model"] == "KBLab/bert-base-swedish-cased"
    assert config["val_file"] == "data/intent_val.jsonl"
    assert config["seed"] == 42


def test_ci_intent_training_config_is_bounded() -> None:
    config = load_config("configs/intent_finetune.ci.yaml")

    assert config["epochs"] == 1
    assert config["max_train_samples"] == 40
    assert config["max_length"] == 64


def test_corpus_sha256_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text("hej\n", encoding="utf-8")

    assert corpus_sha256(path) == corpus_sha256(path)
    assert len(corpus_sha256(path)) == 64
