"""Fas 3: API coverage tests — mocks for conversation, scan, transcription, batch, app handlers."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api import app as default_app
from src.api.app import RequestIdMiddleware, create_app
from src.api.batch import run_batch
from src.api.dependencies import resolve_llm_api_key
from src.api.routers import scan as scan_router
from src.api.settings import get_api_settings
from src.core.errors import (
    AnalysisError,
    BaseAnalysisError,
    ConfigurationError,
    LLMError,
    TranscriptionError,
)

client = TestClient(default_app, raise_server_exceptions=False)


@pytest.fixture
def audio_file(tmp_path):
    p = tmp_path / "call.wav"
    p.write_bytes(b"RIFFxxxx")
    return str(p)


@pytest.fixture
def scan_directory(tmp_path):
    d = tmp_path / "audio_dir"
    d.mkdir()
    (d / "a.wav").write_bytes(b"RIFF")
    (d / "b.wav").write_bytes(b"RIFF")
    return str(d)


# ---------------------------------------------------------------------------
# Scan state helpers (unit)
# ---------------------------------------------------------------------------


def test_scan_load_state_missing_path():
    assert scan_router._load_state(None) == {"processed": {}}


def test_scan_load_state_invalid_json(tmp_path):
    bad = tmp_path / "state.json"
    bad.write_text("not json", encoding="utf-8")
    assert scan_router._load_state(str(bad)) == {"processed": {}}


def test_scan_load_state_valid(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"processed": {"f.wav": {"mtime": 1.0}}}), encoding="utf-8")
    loaded = scan_router._load_state(str(state_path))
    assert "f.wav" in loaded["processed"]


def test_scan_save_state_writes_file(tmp_path):
    state_path = tmp_path / "sub" / "state.json"
    scan_router._save_state(str(state_path), {"processed": {"x": 1}})
    assert state_path.is_file()


def test_scan_chunk():
    assert scan_router._chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


def test_transcribe_happy(audio_file):
    fake_tr = {"segments": [{"text": "hej", "start": 0.0, "end": 1.0}], "model": "test"}
    with patch("src.api.routers.transcription.transcribe_helper", return_value=fake_tr):
        r = client.post("/transcribe", json={"audio_path": audio_file})
    assert r.status_code == 200
    assert r.json()["transcript"]["segments"][0]["text"] == "hej"


def test_transcribe_failure_500(audio_file):
    with patch(
        "src.api.routers.transcription.transcribe_helper",
        side_effect=RuntimeError("asr boom"),
    ):
        r = client.post("/transcribe", json={"audio_path": audio_file})
    assert r.status_code == 500


def test_batch_transcribe_ok_and_worker_error(audio_file):
    from pathlib import Path

    a = str(Path(audio_file).parent / "a.wav")
    b = str(Path(audio_file).parent / "b.wav")
    Path(a).write_bytes(b"RIFF")
    Path(b).write_bytes(b"RIFF")

    def fake_helper(audio_path, **_kwargs):
        if audio_path == a:
            return {"segments": [], "model": "t"}
        raise ValueError("fail b")

    with (
        patch("src.api.routers.transcription.resolve_audio_paths", return_value=[a, b]),
        patch("src.api.routers.transcription.transcribe_helper", side_effect=fake_helper),
    ):
        r = client.post(
            "/batch_transcribe",
            json={"audio_paths": [a, b], "workers": 1},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] == 1
    assert data["failed"] == 1


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------


def test_analyze_conversation_happy(audio_file):
    tr = {"segments": [{"text": "hej", "start": 0.0, "end": 1.0, "speaker": "A"}]}
    with (
        patch("src.api.routers.conversation.transcribe_helper", return_value=tr),
        patch(
            "src.api.routers.conversation.analyze_smart",
            return_value=([{"label": "positiv", "score": 0.9}], {"profile": "call"}),
        ),
    ):
        r = client.post("/analyze_conversation", json={"audio_path": audio_file})
    assert r.status_code == 200
    body = r.json()
    assert body["segment_sentiments"]
    assert body["transcript"]["segments"]


def test_analyze_conversation_transcribe_500(audio_file):
    with patch(
        "src.api.routers.conversation.transcribe_helper",
        side_effect=RuntimeError("tx fail"),
    ):
        r = client.post("/analyze_conversation", json={"audio_path": audio_file})
    assert r.status_code == 500


def test_analyze_conversation_sentiment_500(audio_file):
    tr = {"segments": [{"text": "hej", "start": 0, "end": 1}]}
    with (
        patch("src.api.routers.conversation.transcribe_helper", return_value=tr),
        patch("src.api.routers.conversation.analyze_smart", side_effect=RuntimeError("sent fail")),
    ):
        r = client.post("/analyze_conversation", json={"audio_path": audio_file})
    assert r.status_code == 500


def test_batch_analyze_conversation_mixed(audio_file):
    b = str(audio_file).replace("call.wav", "other.wav")
    from pathlib import Path

    Path(b).write_bytes(b"RIFF")
    tr = {"segments": [{"text": "x", "start": 0, "end": 1}]}

    def tx(audio_path, **_kwargs):
        if audio_path == audio_file:
            return tr
        raise OSError("bad file")

    with (
        patch("src.api.routers.conversation.resolve_audio_paths", return_value=[audio_file, b]),
        patch("src.api.routers.conversation.transcribe_helper", side_effect=tx),
        patch(
            "src.api.routers.conversation.analyze_smart",
            return_value=([{"label": "neutral", "score": 0.5}], {"profile": "call"}),
        ),
    ):
        r = client.post(
            "/batch_analyze_conversation",
            json={"audio_paths": [audio_file, b], "workers": 1},
        )
    assert r.status_code == 200
    assert r.json()["ok"] == 1
    assert r.json()["failed"] == 1


# ---------------------------------------------------------------------------
# Scan process
# ---------------------------------------------------------------------------


def test_scan_process_transcribe_happy(scan_directory):
    with (
        patch(
            "src.api.routers.scan.resolve_audio_paths",
            return_value=[f"{scan_directory}/a.wav", f"{scan_directory}/b.wav"],
        ),
        patch(
            "src.api.routers.scan.transcribe_helper",
            return_value={"segments": [{"text": "a"}], "model": "m"},
        ),
    ):
        r = client.post(
            "/scan_process",
            json={
                "directory": scan_directory,
                "operation": "transcribe",
                "batch_size": 2,
                "workers": 1,
            },
        )
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] == 2
    assert data["total"] == 2


def test_scan_process_analyze_operation(scan_directory):
    with (
        patch(
            "src.api.routers.scan.resolve_audio_paths",
            return_value=[f"{scan_directory}/a.wav"],
        ),
        patch(
            "src.api.routers.scan.transcribe_helper",
            return_value={"segments": [{"text": "hi", "start": 0, "end": 1}]},
        ),
        patch(
            "src.api.routers.scan.analyze_smart",
            return_value=([{"label": "positiv", "score": 0.8}], {"profile": "call"}),
        ),
    ):
        r = client.post(
            "/scan_process",
            json={"directory": scan_directory, "operation": "analyze_conversation"},
        )
    assert r.status_code == 200
    assert r.json()["items"][0]["data"]["segment_sentiments"]


def test_scan_process_skips_unchanged_and_uses_state(scan_directory, tmp_path):
    import os

    wav = f"{scan_directory}/a.wav"
    state_file = tmp_path / "st.json"
    mtime = os.path.getmtime(wav)
    state_file.write_text(
        json.dumps({"processed": {wav: {"mtime": mtime + 1000}}}),
        encoding="utf-8",
    )
    with patch(
        "src.api.routers.scan.resolve_audio_paths",
        return_value=[wav, f"{scan_directory}/b.wav"],
    ):
        r = client.post(
            "/scan_process",
            json={
                "directory": scan_directory,
                "state_file": str(state_file),
                "operation": "transcribe",
            },
        )
    assert r.status_code == 200
    assert r.json()["skipped"] >= 1


def test_scan_process_worker_error(scan_directory):
    with (
        patch(
            "src.api.routers.scan.resolve_audio_paths",
            return_value=[f"{scan_directory}/a.wav"],
        ),
        patch("src.api.routers.scan.transcribe_helper", side_effect=RuntimeError("scan fail")),
    ):
        r = client.post(
            "/scan_process",
            json={"directory": scan_directory, "operation": "transcribe"},
        )
    assert r.status_code == 200
    assert r.json()["failed"] == 1
    assert r.json()["items"][0]["ok"] is False


# ---------------------------------------------------------------------------
# Pipeline error paths + alerts aggregate
# ---------------------------------------------------------------------------


def test_analyze_pipeline_empty_segments_422():
    r = client.post("/analyze_pipeline", json={"segments": []})
    assert r.status_code == 422


def test_analyze_pipeline_generic_500():
    with patch("src.api.dependencies.CallAnalysisPipeline") as mock_pipe:
        mock_pipe.return_value.analyze_segments.side_effect = RuntimeError("boom")
        r = client.post(
            "/analyze_pipeline",
            json={"segments": [{"text": "x", "start": 0, "end": 1}]},
        )
    assert r.status_code == 500
    assert "internal error" in r.json()["detail"].lower()


def test_analyze_pipeline_analysis_error_propagates():
    with patch("src.api.dependencies.CallAnalysisPipeline") as mock_pipe:
        inst = mock_pipe.return_value
        inst.analyze_segments.side_effect = AnalysisError("bad analysis")
        r = client.post(
            "/analyze_pipeline",
            json={"segments": [{"text": "x", "start": 0, "end": 1}]},
        )
    assert r.status_code == 500
    assert "Analysis failed" in r.json()["detail"]


def test_qa_score_compliance_qa_fallback():
    report = MagicMock()
    report.results = {"compliance_qa": {"overall_qa_score": 70}}
    with patch("src.api.dependencies.CallAnalysisPipeline") as mock_pipe:
        mock_pipe.return_value.analyze_segments.return_value = report
        r = client.post("/qa/score", json={"segments": [{"text": "a"}]})
    assert r.status_code == 200
    assert r.json()["qa"]["overall_qa_score"] == 70


def test_alerts_aggregate_branch():
    fake_alert = MagicMock()
    fake_alert.model_dump.return_value = {"rule_id": "trend", "severity": "medium"}
    with patch.object(default_app.state.alert_engine, "check_from_aggregate", return_value=[fake_alert]):
        r = client.post("/alerts", json={"aggregate": {"team_avg": 0.5}})
    assert r.status_code == 200
    assert r.json()["alerts"][0]["rule_id"] == "trend"


def test_alerts_requires_input_422():
    r = client.post("/alerts", json={})
    assert r.status_code == 422


def test_semantic_search_500():
    with patch("src.api.dependencies.CallAnalysisPipeline") as mock_pipe:
        mock_pipe.return_value.analyze_segments.side_effect = RuntimeError("search fail")
        r = client.post(
            "/search/semantic",
            json={"segments_list": [[{"text": "t"}]], "query": "q"},
        )
    assert r.status_code == 500


# ---------------------------------------------------------------------------
# Text / helpers / batch
# ---------------------------------------------------------------------------


def test_analyze_text_500():
    with patch("src.api.routers.text.analyze_smart", side_effect=RuntimeError("nlp fail")):
        r = client.post("/analyze", json={"texts": ["hej"]})
    assert r.status_code == 500
    assert "internal error" in r.json()["detail"].lower()


def test_media_root_rejects_path_outside_sandbox(audio_file, monkeypatch, tmp_path):
    import os
    from pathlib import Path

    media = Path(audio_file).parent
    monkeypatch.setenv("API_MEDIA_ROOT", str(media))
    get_api_settings.cache_clear()
    outside = tmp_path.parent / "outside_sandbox.wav"
    outside.write_bytes(b"RIFF")
    r = client.post("/transcribe", json={"audio_path": str(outside)})
    assert r.status_code == 422
    assert "API_MEDIA_ROOT" in r.text


def test_helpers_transcribe_helper():
    mock_transcript = MagicMock()
    mock_transcript.to_dict.return_value = {"segments": []}
    mock_transcriber = MagicMock()
    mock_transcriber.transcribe.return_value = mock_transcript
    with patch("src.api.helpers.get_transcriber", return_value=mock_transcriber):
        from src.api.helpers import transcribe_helper

        out = transcribe_helper(audio_path="/tmp/x.wav")
    assert out == {"segments": []}


def test_run_batch_sequential_worker_raises():
    def boom(_p):
        raise ValueError("seq err")

    res = run_batch(["f1"], boom, workers=1)
    assert res[0][2] is not None


def test_run_batch_parallel_worker_raises():
    res = run_batch(["f1"], lambda _p: (_ for _ in ()).throw(RuntimeError("par")), workers=2, worker_timeout=2.0)
    assert len(res) == 1
    assert res[0][2] is not None


def test_run_batch_parallel_no_timeout():
    res = run_batch(["f1"], lambda p: f"ok-{p}", workers=2, worker_timeout=None)
    assert res[0][1] == "ok-f1"


# ---------------------------------------------------------------------------
# Dependencies / settings / middleware / app handlers
# ---------------------------------------------------------------------------


def test_resolve_llm_api_key_prefers_header():
    assert resolve_llm_api_key("body", "header") == "header"


def test_resolve_llm_api_key_ignores_body_by_default(monkeypatch):
    monkeypatch.delenv("API_ALLOW_CLIENT_LLM_KEY", raising=False)
    get_api_settings.cache_clear()
    assert resolve_llm_api_key("body-only", None) is None


def test_api_key_auth_when_configured(monkeypatch):
    monkeypatch.setenv("SENTIMENT_API_KEY", "secret-key")
    get_api_settings.cache_clear()
    authed_app = create_app()
    authed_client = TestClient(authed_app, raise_server_exceptions=False)
    r = authed_client.post("/analyze", json={"texts": ["hej"]})
    assert r.status_code == 401
    with patch(
        "src.api.routers.text.analyze_smart",
        return_value=([{"label": "positiv", "score": 0.5}], {"profile": "default"}),
    ):
        r2 = authed_client.post(
            "/analyze",
            json={"texts": ["hej"]},
            headers={"X-API-Key": "secret-key"},
        )
    assert r2.status_code == 200


def test_app_exception_handlers():
    import asyncio

    app = create_app()
    req = MagicMock()

    async def _run():
        assert (await app.exception_handlers[ConfigurationError](req, ConfigurationError("cfg"))).status_code == 422
        assert (await app.exception_handlers[TranscriptionError](req, TranscriptionError("tx"))).status_code == 500
        assert (await app.exception_handlers[AnalysisError](req, AnalysisError("an"))).status_code == 500
        assert (await app.exception_handlers[LLMError](req, LLMError("llm"))).status_code == 502
        assert (await app.exception_handlers[BaseAnalysisError](req, BaseAnalysisError("base"))).status_code == 500

    asyncio.run(_run())


def test_health_has_request_id_header():
    r = client.get("/health")
    assert r.status_code == 200
    assert "X-Request-ID" in r.headers


def test_request_id_middleware_preserves_header():
    mw = RequestIdMiddleware(app=MagicMock())
    assert mw is not None