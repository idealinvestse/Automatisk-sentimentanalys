#!/usr/bin/env python3
"""Run measurable quality gates (intent / Fas4 KPIs / optional sentiment).

Usage:
    python scripts/run_quality_gates.py --smoke
    python scripts/run_quality_gates.py --smoke --min-macro-f1 0.75
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str]) -> int:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    return int(subprocess.call(cmd, cwd=ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Quality KPI gates")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast heuristic-only gates (CI-friendly)",
    )
    parser.add_argument("--min-macro-f1", type=float, default=0.75)
    parser.add_argument(
        "--val-file",
        default="data/intent_val.jsonl",
        help="Intent validation JSONL",
    )
    parser.add_argument(
        "--include-fas4",
        action="store_true",
        help="Also run src.evaluate fas4-validation",
    )
    parser.add_argument(
        "--include-sentiment-model",
        action="store_true",
        help="Heavy: eval_sentiment_gate.py with model backend",
    )
    args = parser.parse_args()

    failures: list[str] = []

    intent_cmd = [
        sys.executable,
        "scripts/benchmark_intent.py",
        "--backend",
        "heuristic",
        "--val-file",
        args.val_file,
        "--min-macro-f1",
        str(args.min_macro_f1),
        "--output",
        "reports/intent_gate_latest.json",
    ]
    if _run(intent_cmd) != 0:
        failures.append("intent_heuristic")

    if (args.include_fas4 or not args.smoke) and _run(
        [sys.executable, "-m", "src.evaluate", "fas4-validation"]
    ) != 0:
        failures.append("fas4_validation")

    if args.include_sentiment_model:
        sent = [
            sys.executable,
            "scripts/eval_sentiment_gate.py",
            "--backend",
            "model",
        ]
        if _run(sent) != 0:
            failures.append("sentiment_model")

    # Persist a short gate summary for ops
    summary = {
        "smoke": args.smoke,
        "min_macro_f1": args.min_macro_f1,
        "failures": failures,
        "ok": not failures,
    }
    out = ROOT / "reports" / "quality_gates_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if failures:
        print(f"\nFAIL: quality gates: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("\nOK: quality gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
