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
