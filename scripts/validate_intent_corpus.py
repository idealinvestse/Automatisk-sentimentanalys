"""Validate intent JSONL corpus balance and deduplication."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.intent import CALL_CENTER_INTENTS


def load_jsonl(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            rows.append({"text": str(item["text"]), "intent": str(item["intent"])})
    return rows


def validate_corpus(
    path: str | Path,
    *,
    min_rows: int = 50,
    min_per_intent: int = 3,
    max_class_ratio: float = 0.25,
    max_duplicate_ratio: float = 0.01,
) -> dict:
    """Validate intent JSONL; raises ValueError on failure."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Corpus not found: {path}")

    rows = load_jsonl(path)
    if len(rows) < min_rows:
        raise ValueError(f"Corpus too small: {len(rows)} rows (min {min_rows})")

    intents = [r["intent"] for r in rows]
    unknown = set(intents) - set(CALL_CENTER_INTENTS)
    if unknown:
        raise ValueError(f"Unknown intents: {sorted(unknown)}")

    dist = Counter(intents)
    for intent in CALL_CENTER_INTENTS:
        if dist.get(intent, 0) < min_per_intent:
            raise ValueError(
                f"Intent '{intent}' has only {dist.get(intent, 0)} rows (min {min_per_intent})"
            )

    max_count = max(dist.values())
    if max_count / len(rows) > max_class_ratio:
        raise ValueError(
            f"Class imbalance: largest class {max_count / len(rows):.1%} "
            f"(max {max_class_ratio:.0%})"
        )

    keys = [(r["text"].lower().strip(), r["intent"]) for r in rows]
    dup_ratio = 1.0 - len(set(keys)) / len(keys)
    if dup_ratio > max_duplicate_ratio:
        raise ValueError(f"Duplicate ratio {dup_ratio:.2%} exceeds max {max_duplicate_ratio:.0%}")

    return {
        "rows": len(rows),
        "intent_distribution": dict(sorted(dist.items())),
        "duplicate_ratio": round(dup_ratio, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate intent JSONL corpus")
    parser.add_argument("path", nargs="?", default="data/intent_train.jsonl")
    parser.add_argument("--min-rows", type=int, default=50)
    parser.add_argument("--min-per-intent", type=int, default=3)
    args = parser.parse_args()
    try:
        stats = validate_corpus(
            args.path,
            min_rows=args.min_rows,
            min_per_intent=args.min_per_intent,
        )
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
    print(
        f"OK: {stats['rows']} rows, dup_ratio={stats['duplicate_ratio']}, "
        f"intents={stats['intent_distribution']}"
    )


if __name__ == "__main__":
    main()
