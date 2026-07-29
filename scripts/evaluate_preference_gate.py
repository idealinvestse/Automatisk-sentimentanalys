#!/usr/bin/env python3
"""Preference-gate evaluate hook (Quality OS scaffolding).

Reads preference pairs JSONL if present; otherwise exits 0 with a clear skip
message so CI can wire the hook without inventing labels (DATA-01).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.quality.mqm import PreferencePair, evaluate_preference_gate  # noqa: E402


def _load_pairs(path: Path) -> list[PreferencePair]:
    if not path.is_file():
        return []
    pairs: list[PreferencePair] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        pairs.append(PreferencePair.model_validate(json.loads(line)))
    return pairs


def _load_gate_config(path: Path) -> dict[str, Any]:
    import yaml

    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data.get("preference_gate", {}) if isinstance(data, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Preference gate for deep-path releases")
    parser.add_argument(
        "--pairs",
        default="data/quality/preference_pairs.jsonl",
        help="JSONL of PreferencePair records",
    )
    parser.add_argument("--config", default="configs/quality_mqm.yaml")
    parser.add_argument("--min-win-rate", type=float, default=None)
    parser.add_argument("--min-pairs", type=int, default=None)
    parser.add_argument(
        "--require-corpus",
        action="store_true",
        help="Exit non-zero when corpus is empty (strict release mode)",
    )
    args = parser.parse_args()

    config = _load_gate_config(Path(args.config))
    min_win_rate = (
        args.min_win_rate
        if args.min_win_rate is not None
        else float(config.get("min_win_rate", 0.55))
    )
    min_pairs = args.min_pairs if args.min_pairs is not None else int(config.get("min_pairs", 1))
    pairs = _load_pairs(Path(args.pairs))
    result = evaluate_preference_gate(pairs, min_win_rate=min_win_rate, min_pairs=min_pairs)

    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))

    if not pairs:
        print(
            "SKIP: no preference corpus — integrate DATA-01 annotations "
            "(configs/quality_mqm.yaml). Scaffold only.",
            file=sys.stderr,
        )
        return 1 if args.require_corpus else 0

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
