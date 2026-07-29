"""Tests for intent corpus validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.validate_intent_corpus import validate_corpus

INTENTS = [
    "account_update",
    "billing_inquiry",
    "technical_support",
    "order_status",
    "cancellation",
    "complaint",
    "information_request",
    "refund_request",
    "appointment_booking",
    "other",
]


def _write_corpus(path: Path, *, overlap: bool = False) -> None:
    rows = [
        {"text": f"exempel {intent} {index}", "intent": intent}
        for intent in INTENTS
        for index in range(3)
    ]
    if overlap:
        rows[0] = {"text": "shared text", "intent": "account_update"}
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_validate_accepts_balanced_corpus(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    _write_corpus(path)

    result = validate_corpus(path, min_rows=10, min_per_intent=3)

    assert result["rows"] == 30
    assert result["duplicate_ratio"] == 0.0


def test_validate_rejects_unknown_intent(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"text": "hej", "intent": "unknown"}), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown intents"):
        validate_corpus(path, min_rows=1, min_per_intent=0)


def test_validate_rejects_overlap_with_other_corpus(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    _write_corpus(train, overlap=True)
    _write_corpus(val, overlap=True)

    with pytest.raises(ValueError, match="overlaps"):
        validate_corpus(train, min_rows=10, min_per_intent=3, disjoint_from=val)
