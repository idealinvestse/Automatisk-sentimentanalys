"""Parametrized 401 matrix for protected API routers."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.settings import get_api_settings

PROTECTED: list[tuple[str, str, dict[str, object] | None]] = [
    ("POST", "/analyze", {"texts": ["hej"]}),
    ("POST", "/transcribe", {"audio_path": "missing.wav"}),
    ("POST", "/scan_process", {"directory": ".", "operation": "transcribe"}),
    ("GET", "/calls", None),
    ("GET", "/llm/providers", None),
    ("POST", "/edge/analyze-text", {"text": "hej"}),
    (
        "POST",
        "/analyze_pipeline",
        {"segments": [{"text": "hej", "speaker": "agent"}]},
    ),
]


@pytest.fixture
def authed_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("SENTIMENT_API_KEY", "secret")
    monkeypatch.delenv("API_MEDIA_ROOT", raising=False)
    get_api_settings.cache_clear()
    return TestClient(create_app(), raise_server_exceptions=False)


@pytest.mark.parametrize(("method", "path", "body"), PROTECTED)
def test_protected_routes_401_without_and_wrong_key(
    authed_client: TestClient,
    method: str,
    path: str,
    body: dict[str, object] | None,
) -> None:
    missing = authed_client.request(method, path, json=body)
    assert missing.status_code == 401, path
    assert missing.json()["error_code"] == "unauthorized"
    assert "request_id" in missing.json()

    wrong = authed_client.request(method, path, json=body, headers={"X-API-Key": "wrong"})
    assert wrong.status_code == 401, path
    assert wrong.json()["error_code"] == "unauthorized"
