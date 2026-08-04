#!/usr/bin/env python3
"""Generate a synthetic DATA-01 pilot-sized corpus into a staging directory.

This does **not** replace a real anonymized telephony corpus. It produces
≥500 sentiment / ≥200 intent rows so CI and local `--pilot-gate` import can
exercise the full DATA-01 path without customer PII.

Usage:
    python scripts/generate_pilot_corpus.py
    python scripts/generate_pilot_corpus.py --import --pilot-gate
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _ensure_base_corpora() -> tuple[Path, Path]:
    sentiment = ROOT / "data" / "callcenter_val.csv"
    intent = ROOT / "data" / "intent_val.jsonl"
    if not sentiment.is_file():
        subprocess.check_call([sys.executable, "scripts/prepare_callcenter_data.py"], cwd=ROOT)
    with sentiment.open(encoding="utf-8") as fh:
        sent_rows = sum(1 for _ in fh) - 1
    if sent_rows < 500:
        subprocess.check_call([sys.executable, "scripts/prepare_callcenter_data.py"], cwd=ROOT)

    intent_rows = (
        sum(1 for line in intent.open(encoding="utf-8") if line.strip()) if intent.is_file() else 0
    )
    if intent_rows < 200:
        # ~10 intents × 120 × 0.25 ≈ 300 val rows (pilot-gate ≥200)
        subprocess.check_call(
            [
                sys.executable,
                "scripts/prepare_intent_data.py",
                "--per-intent",
                "120",
                "--val-ratio",
                "0.25",
            ],
            cwd=ROOT,
        )
    return sentiment, intent


def _copy_named(src: Path, dest_dir: Path, dest_name: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / dest_name
    shutil.copy2(src, dest)
    return dest


def generate(staging_dir: Path) -> dict[str, Path]:
    sentiment_src, intent_src = _ensure_base_corpora()
    # Names recognized by import_domain_corpus.find_source_file
    sent_dest = _copy_named(sentiment_src, staging_dir, "callcenter_val.csv")
    intent_dest = _copy_named(intent_src, staging_dir, "intent_val.jsonl")

    with sent_dest.open(encoding="utf-8") as fh:
        n_sent = sum(1 for _ in csv.DictReader(fh))
    n_intent = sum(1 for line in intent_dest.open(encoding="utf-8") if line.strip())
    if n_sent < 500:
        raise SystemExit(f"Sentiment corpus too small: {n_sent} < 500")
    if n_intent < 200:
        raise SystemExit(f"Intent corpus too small: {n_intent} < 200")

    meta = {
        "kind": "synthetic_pilot_bundle",
        "sentiment_rows": n_sent,
        "intent_rows": n_intent,
        "note": "Synthetic stand-in for DATA-01; replace with anonymized telephony before quality claims",
    }
    (staging_dir / "PILOT_CORPUS_META.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"OK: staged synthetic pilot corpus in {staging_dir}")
    print(f"  sentiment={n_sent} intent={n_intent}")
    return {"sentiment": sent_dest, "intent": intent_dest}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic DATA-01 pilot corpus")
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=ROOT / "data" / "import" / "staging_synthetic",
        help="Directory with callcenter_val.csv + intent_val.jsonl for import",
    )
    parser.add_argument(
        "--import",
        dest="do_import",
        action="store_true",
        help="Also run import_domain_corpus.py into data/import/",
    )
    parser.add_argument("--pilot-gate", action="store_true", help="Pass --pilot-gate to import")
    args = parser.parse_args()

    generate(args.staging_dir)

    if args.do_import:
        cmd = [
            sys.executable,
            "scripts/import_domain_corpus.py",
            "--source-dir",
            str(args.staging_dir),
            "--skip-pii-scan",
        ]
        if args.pilot_gate:
            cmd.append("--pilot-gate")
        print(f">>> {' '.join(cmd)}")
        return int(subprocess.call(cmd, cwd=ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
