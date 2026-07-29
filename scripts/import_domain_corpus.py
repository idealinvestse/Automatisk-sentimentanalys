"""Import anonymized domain corpus from an external directory into the repo import slot.

GDPR: never commit raw customer data. Source files must live outside git; imported
`*_real.*` outputs are gitignored under data/.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.validate_domain_corpus import validate_corpus
from scripts.validate_intent_corpus import validate_corpus as validate_intent_corpus

# Lightweight PII heuristics (subset of src/llm/pii_redactor.py patterns)
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", re.IGNORECASE)
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+46[\s-]?\d{1,4}(?:[\s-]?\d{2,4}){2,4}|0\d{2,3}(?:[\s-]?\d{2,4}){2,4}|\b07\d{8}\b)"
)
_PERSONNUMMER_RE = re.compile(
    r"\b(?:19|20)?\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])[-+]?\d{4}\b"
)

DEFAULT_SENTIMENT_NAMES = ("sentiment.csv", "callcenter_val.csv", "callcenter_sentiment.csv")
DEFAULT_INTENT_NAMES = ("intent.jsonl", "intent_val.jsonl", "callcenter_intent.jsonl")


def scan_pii(text: str) -> list[str]:
    """Return PII pattern names found in text."""
    hits: list[str] = []
    if _EMAIL_RE.search(text):
        hits.append("email")
    if _PHONE_RE.search(text):
        hits.append("phone")
    if _PERSONNUMMER_RE.search(text):
        hits.append("personnummer")
    return hits


def scan_file_pii(path: Path, *, max_lines: int = 500) -> dict[str, int]:
    """Scan first N lines for PII; returns counts by type."""
    counts: dict[str, int] = {}
    with path.open(encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= max_lines:
                break
            for kind in scan_pii(line):
                counts[kind] = counts.get(kind, 0) + 1
    return counts


def find_source_file(source_dir: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = source_dir / name
        if candidate.is_file():
            return candidate
    return None


def import_corpus(
    source_dir: Path,
    *,
    dest_dir: Path,
    min_sentiment_rows: int = 50,
    min_intent_rows: int = 20,
    skip_pii_scan: bool = False,
) -> dict[str, str]:
    """Validate and copy sentiment CSV and/or intent JSONL into data/import/."""
    if not source_dir.is_dir():
        raise ValueError(f"Source directory not found: {source_dir}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    imported: dict[str, str] = {}

    sentiment_src = find_source_file(source_dir, DEFAULT_SENTIMENT_NAMES)
    if sentiment_src:
        if not skip_pii_scan:
            pii = scan_file_pii(sentiment_src)
            if pii:
                raise ValueError(
                    f"PII detected in {sentiment_src.name}: {pii}. "
                    "Anonymize before import or pass --skip-pii-scan after manual review."
                )
        validate_corpus(sentiment_src, min_rows=min_sentiment_rows)
        dest = dest_dir / "callcenter_val_real.csv"
        shutil.copy2(sentiment_src, dest)
        imported["sentiment"] = str(dest)

    intent_src = find_source_file(source_dir, DEFAULT_INTENT_NAMES)
    if intent_src:
        if not skip_pii_scan:
            pii = scan_file_pii(intent_src)
            if pii:
                raise ValueError(
                    f"PII detected in {intent_src.name}: {pii}. "
                    "Anonymize before import or pass --skip-pii-scan after manual review."
                )
        validate_intent_corpus(
            intent_src,
            min_rows=min_intent_rows,
            min_per_intent=2,
        )
        dest = dest_dir / "intent_val_real.jsonl"
        shutil.copy2(intent_src, dest)
        imported["intent"] = str(dest)

    if not imported:
        raise ValueError(
            f"No recognized corpus files in {source_dir}. "
            f"Expected one of sentiment: {DEFAULT_SENTIMENT_NAMES} "
            f"or intent: {DEFAULT_INTENT_NAMES}"
        )
    return imported


def main() -> None:
    parser = argparse.ArgumentParser(description="Import anonymized domain corpus")
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory outside repo with sentiment.csv and/or intent.jsonl",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=Path("data/import"),
        help="Import slot inside repo (gitignored)",
    )
    parser.add_argument("--min-sentiment-rows", type=int, default=50)
    parser.add_argument("--min-intent-rows", type=int, default=20)
    parser.add_argument(
        "--pilot-gate",
        action="store_true",
        help="Enforce pilot corpus minima (500 sentiment / 200 intent rows)",
    )
    parser.add_argument(
        "--skip-pii-scan",
        action="store_true",
        help="Skip automated PII scan (only after manual review)",
    )
    args = parser.parse_args()
    min_sent = 500 if args.pilot_gate else args.min_sentiment_rows
    min_intent = 200 if args.pilot_gate else args.min_intent_rows
    try:
        imported = import_corpus(
            args.source_dir,
            dest_dir=args.dest_dir,
            min_sentiment_rows=min_sent,
            min_intent_rows=min_intent,
            skip_pii_scan=args.skip_pii_scan,
        )
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
    for kind, path in imported.items():
        print(f"OK: imported {kind} -> {path}")
    print(
        "Next: python scripts/evaluate_real_corpus.py --sentiment-csv data/import/callcenter_val_real.csv"
    )


if __name__ == "__main__":
    main()
