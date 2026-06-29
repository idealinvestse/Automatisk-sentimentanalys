"""Tests for evaluate module."""

from __future__ import annotations

import os
import tempfile

import pytest

from src.evaluate import compute_metrics, load_testset, print_results, run_evaluation


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


class TestRunEvaluation:
    def test_heuristic_backend(self, tmp_path):
        csv_path = tmp_path / "tiny.csv"
        csv_path.write_text("text,label\nBra service!,positiv\nUselt!,negativ\n", encoding="utf-8")
        df = load_testset(str(csv_path))
        metrics, details = run_evaluation(df, profile="call", backend="heuristic")
        assert metrics["n_samples"] == 2
        assert metrics["backend"] == "heuristic"
        assert len(details) == 2

    def test_print_results_smoke(self, capsys):
        print_results(
            {
                "accuracy": 0.8,
                "macro_f1": 0.75,
                "model": "test",
                "profile": "call",
                "per_class": {
                    "positiv": {"precision": 1.0, "recall": 0.5, "f1": 0.67, "support": 2},
                    "negativ": {"precision": 0.5, "recall": 1.0, "f1": 0.67, "support": 1},
                    "neutral": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
                },
                "confusion_matrix": {
                    "positiv": {"positiv": 1, "negativ": 0, "neutral": 0},
                    "negativ": {"negativ": 1, "positiv": 0, "neutral": 0},
                    "neutral": {"positiv": 0, "negativ": 0, "neutral": 0},
                },
            }
        )
        captured = capsys.readouterr()
        assert "Accuracy" in captured.out or "accuracy" in captured.out.lower()


class TestEvaluateCLI:
    def test_scenarios_command(self, tmp_path):
        from typer.testing import CliRunner

        from src.evaluate import app

        csv_path = tmp_path / "test.csv"
        csv_path.write_text("text,label\nBra!,positiv\nDåligt!,negativ\n", encoding="utf-8")
        out = tmp_path / "scenarios.json"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "scenarios",
                "--testset",
                str(csv_path),
                "--output",
                str(out),
                "--backend",
                "heuristic",
            ],
        )
        assert result.exit_code == 0
        assert out.is_file()

    def test_asr_compare_command(self, tmp_path):
        from typer.testing import CliRunner

        from src.evaluate import app

        out = tmp_path / "asr.json"
        runner = CliRunner()
        result = runner.invoke(app, ["asr-compare", "--output", str(out)])
        assert result.exit_code == 0
        assert out.is_file()

    def test_llm_quality_command(self, tmp_path):
        from typer.testing import CliRunner

        from src.evaluate import app

        out = tmp_path / "llm_quality.json"
        runner = CliRunner()
        result = runner.invoke(app, ["llm-quality", "--output", str(out)])
        assert result.exit_code == 0
        assert out.is_file()

    def test_negation_command(self, tmp_path):
        from typer.testing import CliRunner

        from src.evaluate import app

        csv_path = tmp_path / "neg.csv"
        csv_path.write_text(
            "text,label\nDet är inte bra.,negativ\nBra!,positiv\n",
            encoding="utf-8",
        )
        out = tmp_path / "negation.json"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "negation",
                "--testset",
                str(csv_path),
                "--output",
                str(out),
                "--backend",
                "heuristic",
            ],
        )
        assert result.exit_code == 0
        assert out.is_file()

    def test_evaluate_command(self, tmp_path):
        from typer.testing import CliRunner

        from src.evaluate import app

        csv_path = tmp_path / "eval.csv"
        csv_path.write_text("text,label\nBra!,positiv\n", encoding="utf-8")
        out = tmp_path / "eval.json"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "evaluate",
                "--testset",
                str(csv_path),
                "--output",
                str(out),
                "--backend",
                "heuristic",
            ],
        )
        assert result.exit_code == 0
        assert out.is_file()

    def test_list_profiles_command(self):
        from typer.testing import CliRunner

        from src.evaluate import app

        runner = CliRunner()
        result = runner.invoke(app, ["list-profiles"])
        assert result.exit_code == 0
        assert "call" in result.output.lower() or "forum" in result.output.lower()
