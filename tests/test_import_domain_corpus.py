"""Tests for the anonymized DATA-01 corpus import workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.import_domain_corpus import import_corpus, scan_pii


def _intent_rows(count: int = 2) -> str:
    intents = [
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
    return "\n".join(
        json.dumps({"text": f"ärende {intent} {index}", "intent": intent})
        for intent in intents
        for index in range(count)
    )


def test_scan_pii_detects_sensitive_patterns() -> None:
    hits = scan_pii("Kontakta anna@example.com eller 0701234567, personnummer 19800101-1234")

    assert set(hits) == {"email", "phone", "personnummer"}


def test_import_rejects_pii(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "intent.jsonl").write_text(
        json.dumps({"text": "anna@example.com", "intent": "other"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="PII detected"):
        import_corpus(source, dest_dir=tmp_path / "dest", min_intent_rows=1)


def test_import_copies_valid_intent_to_real_slot(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "intent.jsonl").write_text(_intent_rows(), encoding="utf-8")

    imported = import_corpus(source, dest_dir=tmp_path / "dest", min_intent_rows=20)

    destination = tmp_path / "dest" / "intent_val_real.jsonl"
    assert imported["intent"] == str(destination)
    assert destination.is_file()
