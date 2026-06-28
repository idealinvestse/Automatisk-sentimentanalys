"""Unified local analyzer benchmarks for CI and offline reports.

Usage:
    python scripts/benchmark_analyzers.py
    python scripts/benchmark_analyzers.py --output reports/analyzer_baseline.json
    python scripts/benchmark_analyzers.py --config configs/analyzer_eval.yaml --check-thresholds
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.benchmark_intent import benchmark_backend, load_intent_jsonl
from src.analysis.compliance_risk import ComplianceRiskAnalyzer
from src.analysis.emotion import EmotionAnalyzer
from src.analysis.negation import NegationAnalyzer
from src.analysis.role_classifier import RoleAnalyzer
from src.core.models import AnalysisContext, Segment
from src.evaluate import run_evaluation

DEFAULT_CONFIG = Path("configs/analyzer_eval.yaml")
DEFAULT_OUTPUT = Path("reports/analyzer_baseline.json")


def load_config(path: Path) -> dict:
    import yaml

    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def benchmark_sentiment(cfg: dict, *, backend: str = "heuristic") -> dict:
    testset = Path(cfg.get("testset", "data/callcenter_val.csv"))
    if not testset.is_file():
        return {"skipped": True, "reason": f"missing {testset}"}
    df = pd.read_csv(testset, encoding="utf-8")
    df["label"] = df["label"].str.strip().str.lower()
    metrics, _ = run_evaluation(
        df,
        profile="callcenter",
        backend=backend,
        batch_size=32,
    )
    return {
        "testset": str(testset),
        "backend": backend,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "per_class": metrics.get("per_class", {}),
        "n_samples": metrics.get("n_samples", len(df)),
    }


def benchmark_intent(cfg: dict) -> dict:
    val_file = Path(cfg.get("val_file", "data/intent_val.jsonl"))
    if not val_file.is_file():
        return {"skipped": True, "reason": f"missing {val_file}"}
    texts, labels = load_intent_jsonl(val_file)
    backend = cfg.get("backend", "heuristic")
    return benchmark_backend(texts, labels, backend=backend)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def benchmark_emotion(path: Path) -> dict:
    if not path.is_file():
        return {"skipped": True, "reason": f"missing {path}"}
    analyzer = EmotionAnalyzer()
    negation = NegationAnalyzer()
    y_true, y_pred = [], []
    for row in _load_jsonl(path):
        seg = Segment(start=0, end=1, text=row["text"], speaker="SPEAKER_0")
        ctx = AnalysisContext(segments=[seg])
        ctx.results["negation"] = negation.analyze(ctx)
        ctx.results["sentiment"] = [{"label": "neutral", "score": 0.5}]
        tl = row["text"].lower()
        if any(w in tl for w in ("arg", "dålig", "katastrof", "väntat", "oacceptabelt")):
            ctx.results["sentiment"] = [{"label": "negativ", "score": 0.85}]
        elif any(w in tl for w in ("perfekt", "nöjd", "tack", "kul", "glad")):
            ctx.results["sentiment"] = [{"label": "positiv", "score": 0.85}]
        out = analyzer.analyze(ctx)[0]
        y_true.append(row["primary"])
        y_pred.append(out["primary"])
    correct = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == p)
    return {
        "fixture": str(path),
        "accuracy": round(correct / max(len(y_true), 1), 4),
        "n_samples": len(y_true),
    }


def benchmark_role(path: Path) -> dict:
    if not path.is_file():
        return {"skipped": True, "reason": f"missing {path}"}
    analyzer = RoleAnalyzer()
    correct_speakers = 0
    total_speakers = 0
    for row in _load_jsonl(path):
        segments = [
            Segment(
                start=float(i),
                end=float(i + 1),
                text=s["text"],
                speaker=s["speaker"],
            )
            for i, s in enumerate(row["segments"])
        ]
        ctx = AnalysisContext(segments=segments, results={"sentiment": []})
        out = analyzer.analyze(ctx)
        roles = out.get("roles", {})
        expected = row["roles"]
        for sp, exp in expected.items():
            total_speakers += 1
            if roles.get(sp) == exp:
                correct_speakers += 1
    return {
        "fixture": str(path),
        "accuracy": round(correct_speakers / max(total_speakers, 1), 4),
        "n_samples": total_speakers,
    }


def benchmark_compliance(path: Path) -> dict:
    if not path.is_file():
        return {"skipped": True, "reason": f"missing {path}"}
    analyzer = ComplianceRiskAnalyzer()
    correct = 0
    total = 0
    for row in _load_jsonl(path):
        seg = Segment(start=0, end=1, text=row["text"], speaker=row["speaker"])
        ctx = AnalysisContext(
            segments=[seg],
            results={"role": {"roles": row.get("roles", {})}},
        )
        out = analyzer.analyze(ctx)
        flagged = out.get("flagged_segments", [])
        got = set(flagged[0]["risks"]) if flagged else set()
        expected = set(row.get("expected_risks") or [])
        total += 1
        if got == expected:
            correct += 1
    return {
        "fixture": str(path),
        "accuracy": round(correct / max(total, 1), 4),
        "n_samples": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark local analyzers")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--sentiment-backend",
        choices=["heuristic", "model"],
        default="heuristic",
        help="Use heuristic in CI for speed",
    )
    parser.add_argument("--check-thresholds", action="store_true")
    args = parser.parse_args()

    if not args.config.is_file():
        raise SystemExit(f"Config not found: {args.config}")

    raw = load_config(args.config)
    report: dict = {
        "config": str(args.config),
        "sentiment": benchmark_sentiment(raw.get("sentiment", {}), backend=args.sentiment_backend),
        "intent": benchmark_intent(raw.get("intent", {})),
    }

    labeled_cfg = raw.get("labeled_fixtures", {})
    report["emotion"] = benchmark_emotion(Path(labeled_cfg.get("emotion", "")))
    report["role"] = benchmark_role(Path(labeled_cfg.get("role", "")))
    report["compliance"] = benchmark_compliance(Path(labeled_cfg.get("compliance", "")))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.check_thresholds:
        errors: list[str] = []
        sent_cfg = raw.get("sentiment", {})
        if not report["sentiment"].get("skipped"):
            if args.sentiment_backend == "heuristic":
                min_acc = sent_cfg.get("heuristic_min_accuracy", 0.4)
                min_f1 = sent_cfg.get("heuristic_min_macro_f1", 0.38)
            else:
                min_acc = sent_cfg.get("min_accuracy", 0.7)
                min_f1 = sent_cfg.get("min_macro_f1", 0.68)
            if report["sentiment"]["accuracy"] < min_acc:
                errors.append("sentiment accuracy below threshold")
            if report["sentiment"]["macro_f1"] < min_f1:
                errors.append("sentiment macro_f1 below threshold")
        intent_cfg = raw.get("intent", {})
        if not report["intent"].get("skipped"):
            if report["intent"]["f1_macro"] < intent_cfg.get("min_macro_f1", 0):
                errors.append("intent macro_f1 below threshold")
            if report["intent"]["accuracy"] < intent_cfg.get("min_accuracy", 0):
                errors.append("intent accuracy below threshold")
        for name in ("emotion", "role", "compliance"):
            block = report.get(name, {})
            if not block.get("skipped") and block.get("accuracy", 0) < labeled_cfg.get(
                "min_accuracy", 0.6
            ):
                errors.append(f"{name} accuracy below threshold")
        if errors:
            for e in errors:
                print(f"FAIL: {e}", file=sys.stderr)
            raise SystemExit(1)


if __name__ == "__main__":
    main()
