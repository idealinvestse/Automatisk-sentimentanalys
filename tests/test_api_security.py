"""Security hardening tests for the REST API."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api import app as default_app
from src.api.app import create_app
from src.api.schemas import MAX_ANALYZE_TEXTS
from src.api.settings import get_api_settings


@pytest.fixture(autouse=True)
def _clear_api_settings_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SENTIMENT_API_KEY", raising=False)
    monkeypatch.delenv("API_MEDIA_ROOT", raising=False)
    monkeypatch.delenv("API_STATE_DIR", raising=False)
    get_api_settings.cache_clear()


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


def test_batch_transcribe_rejects_path_outside_media_root(tmp_path, monkeypatch):
    media = tmp_path / "media"
    media.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"RIFF")
    monkeypatch.setenv("API_MEDIA_ROOT", str(media))
    get_api_settings.cache_clear()
    r = client.post("/batch_transcribe", json={"audio_paths": [str(outside)], "workers": 1})
    assert r.status_code == 422
    assert "API_MEDIA_ROOT" in r.text


def test_batch_transcribe_rejects_directory_outside_media_root(tmp_path, monkeypatch):
    media = tmp_path / "media"
    media.mkdir()
    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir()
    monkeypatch.setenv("API_MEDIA_ROOT", str(media))
    get_api_settings.cache_clear()
    r = client.post("/batch_transcribe", json={"directory": str(outside_dir), "workers": 1})
    assert r.status_code == 422
    assert "API_MEDIA_ROOT" in r.text


def test_batch_transcribe_requires_source():
    r = client.post("/batch_transcribe", json={"workers": 1})
    assert r.status_code == 422
    assert "audio_paths or directory" in r.text.lower()


def test_scan_process_rejects_state_file_outside_state_dir(scan_directory, tmp_path):
    state_file = tmp_path / "escape" / "state.json"
    r = client.post(
        "/scan_process",
        json={
            "directory": scan_directory,
            "operation": "transcribe",
            "state_file": str(state_file),
        },
    )
    assert r.status_code == 422
    assert "state" in r.text.lower()


def test_scan_process_accepts_state_file_under_state_dir(scan_directory, tmp_path, monkeypatch):
    monkeypatch.setenv("API_STATE_DIR", str(tmp_path))
    get_api_settings.cache_clear()
    state_file = tmp_path / "state.json"
    with (
        patch(
            "src.api.routers.scan.resolve_and_validate_audio_paths",
            return_value=[f"{scan_directory}/a.wav"],
        ),
        patch(
            "src.api.routers.scan.transcribe_helper",
            return_value={"segments": [], "model": "m"},
        ),
    ):
        r = client.post(
            "/scan_process",
            json={
                "directory": scan_directory,
                "operation": "transcribe",
                "state_file": str(state_file),
            },
        )
    assert r.status_code == 200
    assert state_file.is_file()


def test_resolved_symlink_outside_media_root_rejected(monkeypatch, tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    outside = tmp_path / "secret.wav"
    outside.write_bytes(b"RIFF")
    link = media / "linked.wav"
    link.symlink_to(outside)
    monkeypatch.setenv("API_MEDIA_ROOT", str(media))
    get_api_settings.cache_clear()
    r = client.post("/transcribe", json={"audio_path": str(link)})
    assert r.status_code == 422
    assert "API_MEDIA_ROOT" in r.text


def test_analyze_rejects_too_many_texts():
    texts = ["hej"] * (MAX_ANALYZE_TEXTS + 1)
    r = client.post("/analyze", json={"texts": texts})
    assert r.status_code == 422
    assert str(MAX_ANALYZE_TEXTS) in r.text


def test_status_endpoints_require_api_key_when_configured(monkeypatch):
    monkeypatch.setenv("SENTIMENT_API_KEY", "secret-key")
    get_api_settings.cache_clear()
    authed_client = TestClient(create_app(), raise_server_exceptions=False)
    assert authed_client.get("/status/processes").status_code == 401
    assert authed_client.get("/status/health/detail").status_code == 401
    assert authed_client.get("/metrics").status_code == 401
    assert (
        authed_client.get("/status/processes", headers={"X-API-Key": "secret-key"}).status_code
        == 200
    )


def test_rate_limit_ignores_forwarded_for_without_trusted_proxy(monkeypatch):
    monkeypatch.setenv("API_RATE_LIMIT_RPM", "2")
    monkeypatch.delenv("API_TRUSTED_PROXY", raising=False)
    get_api_settings.cache_clear()
    limited_client = TestClient(create_app(), raise_server_exceptions=False)
    for i in range(3):
        r = limited_client.get("/status/processes", headers={"X-Forwarded-For": f"1.2.3.{i}"})
    assert r.status_code == 429


def test_rate_limit_honors_forwarded_for_with_trusted_proxy(monkeypatch):
    monkeypatch.setenv("API_RATE_LIMIT_RPM", "2")
    monkeypatch.setenv("API_TRUSTED_PROXY", "true")
    get_api_settings.cache_clear()
    limited_client = TestClient(create_app(), raise_server_exceptions=False)
    headers = {"X-Forwarded-For": "1.2.3.4"}
    for _ in range(2):
        assert limited_client.get("/status/processes", headers=headers).status_code == 200
    r = limited_client.get("/status/processes", headers=headers)
    assert r.status_code == 429


def test_metrics_use_route_template_not_raw_path(monkeypatch):
    monkeypatch.delenv("SENTIMENT_API_KEY", raising=False)
    get_api_settings.cache_clear()
    with patch("src.api.dependencies.CallAnalysisPipeline") as mock_pipe:
        inst = mock_pipe.return_value
        inst.analyze_segments.return_value = __import__(
            "unittest.mock", fromlist=["MagicMock"]
        ).MagicMock(
            sentiment_results=[],
            intent_results=[],
            summary={},
            topics={},
            insights={},
            risks={},
            processing_time_s=0.1,
            llm={},
            results={},
        )
        inst.get_cached_agent_performance.return_value = {"call_count": 1}
        client.post(
            "/agent_performance/Agent-123",
            json={"segments_list": [[{"text": "Hej"}]], "agent_id": "Agent-123"},
        )
    metrics = client.get("/metrics").text
    if "prometheus_client not installed" in metrics:
        pytest.skip("prometheus_client not installed")
    assert 'path="/agent_performance/{agent_id}"' in metrics
    assert 'path="/agent_performance/Agent-123"' not in metrics
