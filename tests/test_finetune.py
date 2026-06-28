"""Tests for fine-tuning config/data helpers."""

from __future__ import annotations

import os
import tempfile

import pytest

from src.finetune import LABEL_TO_ID, FinetuneConfig, load_labelled_csv


def test_label_mapping():
    assert LABEL_TO_ID == {"negativ": 0, "neutral": 1, "positiv": 2}


def test_load_labelled_csv_valid():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write("text,label\nTack,positiv\nVäntar,neutral\nDåligt,negativ\n")
        path = f.name
    try:
        df = load_labelled_csv(path)
        assert len(df) == 3
    finally:
        os.unlink(path)


def test_load_labelled_csv_rejects_unknown_label():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write("text,label\nHej,okänd\n")
        path = f.name
    try:
        with pytest.raises(ValueError, match="Unknown labels"):
            load_labelled_csv(path)
    finally:
        os.unlink(path)


def test_finetune_config_defaults():
    cfg = FinetuneConfig(
        model_name="KB/bert-base-swedish-cased",
        train_file="train.csv",
        eval_file="eval.csv",
        output_dir="out",
    )
    assert cfg.max_length == 256
    assert cfg.num_train_epochs == 3


def test_load_finetune_ci_config():
    from src.finetune import load_config

    cfg = load_config("configs/finetune.ci.yaml")
    assert cfg.num_train_epochs == 1
    assert cfg.per_device_train_batch_size == 4


def test_callcenter_val_corpus_size():
    from pathlib import Path

    from src.finetune import load_labelled_csv

    path = Path("data/callcenter_val.csv")
    if not path.is_file():
        pytest.skip("callcenter_val.csv not present")
    df = load_labelled_csv(str(path))
    assert len(df) >= 500


def test_finetune_baseline_file():
    import json
    from pathlib import Path

    path = Path("reports/finetune_baseline.json")
    if not path.is_file():
        pytest.skip("finetune baseline missing")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "accuracy" in data
    assert "f1_macro" in data
    assert data["regression_tolerance"] >= 0


def test_callcenter_profile_uses_lora_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    from src import profiles

    monkeypatch.setattr(profiles.os.path, "isdir", lambda p: str(p).endswith("callcenter-sentiment-lora"))
    assert "callcenter-sentiment-lora" in profiles._get_callcenter_model()


def test_intent_model_path_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENT_MODEL_PATH", "models/custom_intent")
    from importlib import reload

    import src.analysis.resources as res

    reload(res)
    assert res.DEFAULT_INTENT_MODEL_PATH == "models/custom_intent"
