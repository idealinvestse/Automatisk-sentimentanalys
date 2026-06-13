"""Unit tests for scan_process state helpers (Fas 5)."""

from __future__ import annotations

from src.api.routers import scan as scan_router


def test_load_state_missing_returns_empty():
    assert scan_router._load_state(None) == {"processed": {}}


def test_save_and_load_state_roundtrip(tmp_path):
    path = str(tmp_path / "state.json")
    state = {"processed": {"/a.wav": {"mtime": 1.0, "when": "2026-01-01T00:00:00Z"}}}
    scan_router._save_state(path, state)
    loaded = scan_router._load_state(path)
    assert loaded["processed"]["/a.wav"]["mtime"] == 1.0


def test_load_state_invalid_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("not json", encoding="utf-8")
    assert scan_router._load_state(str(path)) == {"processed": {}}


def test_chunk_splits_evenly():
    assert scan_router._chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
