"""GDPR-safe domain corpus import — thin wrapper over ``scripts.import_domain_corpus``.

Prefer calling ``python scripts/import_domain_corpus.py`` from the CLI.
This module exists so library code can import the same path without duplicating logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def gdpr_safe_import(
    source_dir: str | Path,
    *,
    dest_dir: str | Path = "data/import",
    pilot_gate: bool = False,
    skip_pii_scan: bool = False,
) -> dict[str, str]:
    """Validate, PII-scan, and copy anonymized corpus into the import slot."""
    from scripts.import_domain_corpus import import_corpus

    min_sent = 500 if pilot_gate else 50
    min_intent = 200 if pilot_gate else 20
    return import_corpus(
        Path(source_dir),
        dest_dir=Path(dest_dir),
        min_sentiment_rows=min_sent,
        min_intent_rows=min_intent,
        skip_pii_scan=skip_pii_scan,
    )


def import_status(dest_dir: str | Path = "data/import") -> dict[str, Any]:
    """Return which import-slot files exist and rough row counts."""
    root = Path(dest_dir)
    out: dict[str, Any] = {"dir": str(root), "files": {}}
    for name in ("callcenter_val_real.csv", "intent_val_real.jsonl"):
        path = root / name
        if not path.is_file():
            out["files"][name] = {"exists": False}
            continue
        if name.endswith(".csv"):
            rows = max(0, sum(1 for _ in path.open(encoding="utf-8")) - 1)
        else:
            rows = sum(1 for line in path.open(encoding="utf-8") if line.strip())
        out["files"][name] = {"exists": True, "rows": rows}
    return out
