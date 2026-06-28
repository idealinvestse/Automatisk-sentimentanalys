"""Compare sentiment evaluation metrics against finetune baseline (DATA-01 gate)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import run_evaluation

DEFAULT_BASELINE = Path("reports/finetune_baseline.json")
DEFAULT_TESTSET = Path("data/callcenter_val.csv")


def load_baseline(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentiment eval vs finetune baseline")
    parser.add_argument("--testset", type=Path, default=DEFAULT_TESTSET)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--profile", default="callcenter")
    parser.add_argument("--backend", choices=["heuristic", "model"], default="model")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not args.testset.is_file():
        raise SystemExit(f"Testset not found: {args.testset}")
    if not args.baseline.is_file():
        raise SystemExit(f"Baseline not found: {args.baseline}")

    baseline = load_baseline(args.baseline)
    tol = float(baseline.get("regression_tolerance", 0.02))
    if args.backend == "heuristic":
        min_acc = float(baseline.get("heuristic_min_accuracy", 0.40))
        min_f1 = float(baseline.get("heuristic_min_macro_f1", 0.38))
    else:
        min_acc = float(baseline.get("accuracy", 0.72)) - tol
        min_f1 = float(baseline.get("f1_macro", 0.70)) - tol

    df = pd.read_csv(args.testset, encoding="utf-8")
    df["label"] = df["label"].str.strip().str.lower()
    metrics, _ = run_evaluation(df, profile=args.profile, backend=args.backend, batch_size=32)

    report = {
        "testset": str(args.testset),
        "backend": args.backend,
        "profile": args.profile,
        "metrics": metrics,
        "baseline": baseline,
        "gates": {"min_accuracy": min_acc, "min_macro_f1": min_f1},
        "passed": metrics["accuracy"] >= min_acc and metrics["macro_f1"] >= min_f1,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        print(
            f"FAIL: accuracy {metrics['accuracy']} (min {min_acc}), "
            f"macro_f1 {metrics['macro_f1']} (min {min_f1})",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
