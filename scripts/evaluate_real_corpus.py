"""Offline evaluation for anonymized real corpora (GDPR — never commit raw data)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._bootstrap import bootstrap_script
from scripts.benchmark_intent import benchmark_backend, load_intent_jsonl
from scripts.validate_domain_corpus import validate_corpus
from scripts.validate_intent_corpus import validate_corpus as validate_intent_corpus
from src.evaluate import run_evaluation

DEFAULT_BASELINE = Path("reports/domain_baseline.json")


def _load_sentiment_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    df["label"] = df["label"].str.strip().str.lower()
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate external anonymized corpus (outputs to reports/offline/)"
    )
    parser.add_argument(
        "--sentiment-csv",
        type=Path,
        default=None,
        help="CSV with text,label (negativ/neutral/positiv)",
    )
    parser.add_argument(
        "--intent-jsonl",
        type=Path,
        default=None,
        help="JSONL with text,intent",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/offline"),
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--profile", default="callcenter")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    bootstrap_script(verbose=args.verbose)

    if not args.sentiment_csv and not args.intent_jsonl:
        raise SystemExit("Provide --sentiment-csv and/or --intent-jsonl")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report: dict = {"timestamp": stamp, "sentiment": None, "intent": None}

    if args.sentiment_csv:
        if not args.sentiment_csv.is_file():
            raise SystemExit(f"File not found: {args.sentiment_csv}")
        validate_corpus(args.sentiment_csv, min_rows=50)
        df = _load_sentiment_csv(args.sentiment_csv)
        metrics, _ = run_evaluation(df, profile=args.profile, backend="model", batch_size=32)
        report["sentiment"] = {
            "source": str(args.sentiment_csv),
            "metrics": metrics,
        }

    if args.intent_jsonl:
        if not args.intent_jsonl.is_file():
            raise SystemExit(f"File not found: {args.intent_jsonl}")
        validate_intent_corpus(args.intent_jsonl, min_rows=20, min_per_intent=2)
        texts, labels = load_intent_jsonl(args.intent_jsonl)
        report["intent"] = {
            "source": str(args.intent_jsonl),
            "heuristic": benchmark_backend(texts, labels, backend="heuristic"),
        }

    if args.baseline.is_file():
        report["domain_baseline_reference"] = json.loads(args.baseline.read_text(encoding="utf-8"))

    out_path = args.output_dir / f"real_corpus_eval_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Report written: {out_path}")


if __name__ == "__main__":
    main()
