"""Tests for server-side call persistence."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.call_store import CallStore
from src.api.settings import get_api_settings


def test_call_store_roundtrip(tmp_path) -> None:
    store = CallStore(tmp_path)
    doc = store.save(
        "call-1",
        {"transcript": {"id": "call-1", "title": "Test"}, "report": {"mode": "full"}},
    )
    assert doc["id"] == "call-1"
    assert store.get("call-1")["transcript"]["title"] == "Test"
    listed = store.list(limit=10)
    assert len(listed) == 1
    assert store.delete("call-1") is True
    assert store.get("call-1") is None


def test_calls_api_crud(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("SENTIMENT_API_KEY", raising=False)
    monkeypatch.setenv("API_STATE_DIR", str(tmp_path))
    get_api_settings.cache_clear()
    client = TestClient(create_app())

    created = client.post(
        "/calls",
        json={
            "id": "abc-123",
            "transcript": {"id": "abc-123", "title": "Faktura"},
            "report": {"degraded": [], "mode": "full"},
        },
    )
    assert created.status_code == 200
    assert created.json()["id"] == "abc-123"

    listed = client.get("/calls?limit=10")
    assert listed.status_code == 200
    assert listed.json()["count"] >= 1

    got = client.get("/calls/abc-123")
    assert got.status_code == 200
    assert got.json()["transcript"]["title"] == "Faktura"

    deleted = client.delete("/calls/abc-123")
    assert deleted.status_code == 200
    assert client.get("/calls/abc-123").status_code == 404


def test_calls_put_upsert_and_path_body_mismatch(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("SENTIMENT_API_KEY", raising=False)
    monkeypatch.setenv("API_STATE_DIR", str(tmp_path))
    get_api_settings.cache_clear()
    client = TestClient(create_app())
    payload = {
        "id": "call-put",
        "transcript": {"id": "call-put", "title": "Uppdaterad"},
        "report": {"mode": "full"},
    }
    created = client.put("/calls/call-put", json=payload)
    assert created.status_code == 200
    assert created.json()["transcript"]["title"] == "Uppdaterad"

    mismatch = client.put("/calls/other-id", json=payload)
    assert mismatch.status_code == 422
    body = mismatch.json()
    assert body["error_code"] == "validation_error"
    assert "match" in str(body["detail"]).lower()


def test_call_store_invalid_id_and_corrupt_files(tmp_path) -> None:
    store = CallStore(tmp_path)
    with pytest.raises(ValueError, match="Invalid call id"):
        store.save("../evil", {"transcript": {}})
    (tmp_path / "calls" / "broken.json").write_text("{not-json", encoding="utf-8")
    assert store.get("broken") is None
    assert store.list(limit=10) == []
    assert store.delete("missing") is False


def test_calls_delete_missing_and_lazy_store(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("SENTIMENT_API_KEY", raising=False)
    monkeypatch.setenv("API_STATE_DIR", str(tmp_path))
    get_api_settings.cache_clear()
    client = TestClient(create_app())
    if hasattr(client.app.state, "call_store"):
        delattr(client.app.state, "call_store")
    listed = client.get("/calls")
    assert listed.status_code == 200
    missing = client.delete("/calls/does-not-exist")
    assert missing.status_code == 404
