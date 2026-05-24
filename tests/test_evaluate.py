"""Tests for evaluate module."""
from __future__ import annotations

import os
import tempfile

import pytest

from src.evaluate import compute_metrics, load_testset


class TestLoadTestset:
    def test_valid_csv(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write("text,label,source\nBra!,positiv,test\nDåligt!,negativ,test\n")
            path = f.name
        try:
            df = load_testset(path)
            assert len(df) == 2
            assert df.iloc[0]["label"] == "positiv"
            assert df.iloc[1]["label"] == "negativ"
        finally:
            os.unlink(path)

    def test_missing_column(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write("col1,col2\na,b\n")
            path = f.name
        try:
            with pytest.raises(ValueError, match="text"):
                load_testset(path)
        finally:
            os.unlink(path)


class TestComputeMetrics:
    def test_perfect_predictions(self):
        y_true = ["positiv", "negativ", "neutral"]
        y_pred = ["positiv", "negativ", "neutral"]
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0

    def test_all_wrong(self):
        y_true = ["positiv", "negativ", "neutral"]
        y_pred = ["negativ", "positiv", "positiv"]
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 0.0
        assert metrics["macro_f1"] < 0.5

    def test_per_class_metrics(self):
        y_true = ["positiv", "positiv", "negativ"]
        y_pred = ["positiv", "negativ", "negativ"]
        metrics = compute_metrics(y_true, y_pred)
        pc = metrics["per_class"]
        assert pc["positiv"]["support"] == 2
        assert pc["negativ"]["support"] == 1
        assert pc["neutral"]["support"] == 0
        # positiv: TP=1, FP=0, FN=1 -> precision=1.0, recall=0.5
        assert pc["positiv"]["precision"] == 1.0
        assert pc["positiv"]["recall"] == 0.5

    def test_confusion_matrix(self):
        y_true = ["positiv", "negativ"]
        y_pred = ["positiv", "negativ"]
        metrics = compute_metrics(y_true, y_pred)
        cm = metrics["confusion_matrix"]
        assert cm["positiv"]["positiv"] == 1
        assert cm["negativ"]["negativ"] == 1
        assert cm["positiv"]["negativ"] == 0

    def test_empty(self):
        metrics = compute_metrics([], [])
        assert metrics["n_samples"] == 0
        assert metrics["accuracy"] == 0.0
