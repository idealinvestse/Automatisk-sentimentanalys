"""Tests for server-side call persistence."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.call_persistence import persist_call_artifact, persist_intake_file
from src.api.call_store import CallStore, call_idempotency_key, new_call_id
from src.api.settings import get_api_settings
from src.customers import CustomerAsrPolicy, CustomerContext, CustomerLlmPolicy


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


def test_calls_api_issues_server_id_when_omitted(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("SENTIMENT_API_KEY", raising=False)
    monkeypatch.setenv("API_STATE_DIR", str(tmp_path))
    get_api_settings.cache_clear()
    client = TestClient(create_app())
    created = client.post(
        "/calls",
        json={"transcript": {"title": "Ny"}, "report": {"mode": "full"}},
    )
    assert created.status_code == 200
    issued = created.json()["id"]
    assert issued
    assert issued != "Ny"
    assert client.get(f"/calls/{issued}").status_code == 200


def test_call_store_idempotency_and_customer_fields(tmp_path) -> None:
    store = CallStore(tmp_path)
    key = call_idempotency_key("0042", "kund-0042_a.wav", "abc123")
    first = store.save(
        new_call_id(),
        {
            "status": "transcribed",
            "customer_id": "0042",
            "idempotency_key": key,
            "transcript": {"segments": [{"text": "hej"}]},
            "provenance": {"original_filename": "kund-0042_a.wav", "route": "transcribe"},
        },
    )
    found = store.find_by_idempotency(key)
    assert found is not None
    assert found["id"] == first["id"]
    assert found["customer_id"] == "0042"
    updated = store.save(
        first["id"],
        {
            "status": "completed",
            "report": {"mode": "full"},
            "idempotency_key": key,
            "customer_id": "0042",
        },
    )
    assert updated["id"] == first["id"]
    assert updated["status"] == "completed"
    assert updated["transcript"]["segments"][0]["text"] == "hej"
    assert call_idempotency_key(None, "", None) is None
    first_anon = store.save(new_call_id(), {"status": "completed", "transcript": {"n": 1}})
    second_anon = store.save(new_call_id(), {"status": "completed", "transcript": {"n": 2}})
    assert first_anon["id"] != second_anon["id"]


def test_find_by_idempotency_beyond_list_window(tmp_path) -> None:
    store = CallStore(tmp_path)
    key = call_idempotency_key("0042", "old.wav", "fp-old")
    old = store.save(
        new_call_id(),
        {"status": "transcribed", "customer_id": "0042", "idempotency_key": key},
    )
    for i in range(500):
        store.save(
            f"extra-{i:04d}",
            {"status": "completed", "idempotency_key": f"other-{i}"},
        )
    old_path = tmp_path / "calls" / f"{old['id']}.json"
    old_path.touch()
    os.utime(old_path, (1_000_000, 1_000_000))
    newest = {doc["id"] for doc in store.list(limit=500)}
    assert old["id"] not in newest
    found = store.find_by_idempotency(key)
    assert found is not None
    assert found["id"] == old["id"]


def test_persist_call_artifact_idempotent_and_transcript_before_report(tmp_path) -> None:
    store = CallStore(tmp_path)
    customer = CustomerContext(
        customer_id="0042",
        display_name="Testkund AB",
        analyzer_profile="callcenter",
        qa_scorecard="standard_support_v1",
        asr=CustomerAsrPolicy(),
        llm=CustomerLlmPolicy(),
        registry_version=1,
        config_fingerprint="fp-1",
        source_filename="kund-0042_a.wav",
    )
    first = persist_call_artifact(
        store,
        status="transcribed",
        transcript={"segments": [{"text": "hej"}]},
        customer=customer,
        original_filename="kund-0042_a.wav",
        route="transcribe",
    )
    second = persist_call_artifact(
        store,
        status="completed",
        report={"mode": "full"},
        customer=customer,
        original_filename="kund-0042_a.wav",
        route="analyze_pipeline",
    )
    assert first["id"] == second["id"]
    assert second["status"] == "completed"
    assert second["transcript"]["segments"][0]["text"] == "hej"
    assert second["report"]["mode"] == "full"
    assert second["customer_id"] == "0042"
    assert second["meta"]["customer"]["config_fingerprint"] == "fp-1"
    assert second["provenance"]["original_filename"] == "kund-0042_a.wav"
    assert second["provenance"]["config_fingerprint"] == "fp-1"
    other = persist_call_artifact(
        store,
        status="completed",
        transcript={"segments": [{"text": "annan"}]},
        customer=customer.model_copy(update={"config_fingerprint": "fp-2"}),
        original_filename="kund-0042_a.wav",
        route="analyze_pipeline",
    )
    assert other["id"] != first["id"]
    failed = persist_call_artifact(
        store,
        status="failed",
        customer=customer,
        original_filename="kund-0042_b.wav",
        route="transcribe",
        fail_reason="asr_empty_transcript",
    )
    assert failed["status"] == "failed"
    assert failed["provenance"]["fail_reason"] == "asr_empty_transcript"


def test_call_store_merges_meta_without_dropping_customer(tmp_path) -> None:
    store = CallStore(tmp_path)
    call_id = new_call_id()
    store.save(
        call_id,
        {"meta": {"customer": {"customer_id": "0042"}}, "customer_id": "0042"},
    )
    updated = store.save(call_id, {"meta": {"source": "webui"}})
    assert updated["meta"]["customer"]["customer_id"] == "0042"
    assert updated["meta"]["source"] == "webui"
    assert updated["customer_id"] == "0042"


def test_persist_intake_file_skips_missing_store(tmp_path) -> None:
    assert persist_intake_file(None, audio_path="x.wav", route="batch_transcribe", status="failed") is None
    store = CallStore(tmp_path)
    doc = persist_intake_file(
        store,
        audio_path="kund-demo.wav",
        route="batch_transcribe",
        status="transcribed",
        transcript={"segments": [{"text": "hej"}]},
    )
    assert doc is not None
    assert doc["provenance"]["route"] == "batch_transcribe"
    assert doc["provenance"]["original_filename"] == "kund-demo.wav"


def test_persist_intake_file_completed_raises_on_store_error(tmp_path) -> None:
    store = CallStore(tmp_path)
    with (
        patch.object(store, "save", side_effect=OSError("disk full")),
        pytest.raises(OSError, match="disk full"),
    ):
        persist_intake_file(
            store,
            audio_path="kund-demo.wav",
            route="batch_transcribe",
            status="transcribed",
            transcript={"segments": [{"text": "hej"}]},
        )


def test_persist_intake_file_failed_is_best_effort(tmp_path) -> None:
    store = CallStore(tmp_path)
    with patch.object(store, "save", side_effect=OSError("disk full")):
        assert (
            persist_intake_file(
                store,
                audio_path="kund-demo.wav",
                route="batch_transcribe",
                status="failed",
                must_succeed=False,
            )
            is None
        )


def test_batch_transcribe_persists_ok_and_failed(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("SENTIMENT_API_KEY", raising=False)
    monkeypatch.setenv("API_STATE_DIR", str(tmp_path))
    get_api_settings.cache_clear()
    client = TestClient(create_app())
    ok_path = str(tmp_path / "ok.wav")
    bad_path = str(tmp_path / "bad.wav")

    def fake_helper(audio_path, **_kwargs):
        if audio_path == ok_path:
            return {"segments": [{"text": "hej"}], "model": "t"}
        raise ValueError("fail b")

    with (
        patch(
            "src.api.routers.transcription.resolve_and_validate_audio_paths",
            return_value=[ok_path, bad_path],
        ),
        patch("src.api.routers.transcription.transcribe_helper", side_effect=fake_helper),
    ):
        r = client.post(
            "/batch_transcribe",
            json={"audio_paths": [ok_path, bad_path], "workers": 1},
        )
    assert r.status_code == 200
    assert r.json()["ok"] == 1
    assert r.json()["failed"] == 1
    store = CallStore(tmp_path)
    docs = store.list(limit=20)
    statuses = {doc.get("status") for doc in docs}
    assert "transcribed" in statuses
    assert "failed" in statuses
    routes = {doc.get("provenance", {}).get("route") for doc in docs}
    assert "batch_transcribe" in routes
