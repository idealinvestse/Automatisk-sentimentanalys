"""Validate domain corpus size and label distribution."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from src.finetune import LABEL_TO_ID, load_labelled_csv


def validate_corpus(
    path: str | Path,
    *,
    min_rows: int = 500,
    max_unknown_ratio: float = 0.0,
) -> dict:
    """Validate labelled CSV; raises ValueError on failure."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Corpus not found: {path}")

    df = load_labelled_csv(str(path))
    if len(df) < min_rows:
        raise ValueError(f"Corpus too small: {len(df)} rows (min {min_rows})")

    dist = Counter(df["label"].tolist())
    unknown = set(dist) - set(LABEL_TO_ID)
    if unknown:
        raise ValueError(f"Unknown labels: {sorted(unknown)}")

    return {"rows": len(df), "label_distribution": dict(dist)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate domain sentiment corpus")
    parser.add_argument("path", nargs="?", default="data/callcenter_val.csv")
    parser.add_argument("--min-rows", type=int, default=500)
    args = parser.parse_args()
    try:
        stats = validate_corpus(args.path, min_rows=args.min_rows)
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"OK: {stats['rows']} rows, labels={stats['label_distribution']}")


if __name__ == "__main__":
    main()
