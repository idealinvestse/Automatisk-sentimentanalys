"""Contract tests: structured API error body shape (Fas 5)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.settings import get_api_settings


@pytest.fixture(autouse=True)
def _clear_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SENTIMENT_API_KEY", raising=False)
    monkeypatch.delenv("API_RATE_LIMIT_RPM", raising=False)
    get_api_settings.cache_clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(), raise_server_exceptions=False)


def test_error_contract_validation(client: TestClient) -> None:
    r = client.post("/analyze", json={"texts": []})
    assert r.status_code == 422
    body = r.json()
    assert "detail" in body
    assert body["error_code"] == "validation_error"
    assert "request_id" in body
    assert r.headers["X-Request-ID"] == body["request_id"]


def test_error_contract_internal(client: TestClient) -> None:
    with patch("src.api.routers.text.analyze_smart", side_effect=RuntimeError("boom")):
        r = client.post("/analyze", json={"texts": ["hej"]})
    assert r.status_code == 500
    body = r.json()
    assert body["error_code"] == "internal_error"
    assert "request_id" in body


def test_error_contract_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_RATE_LIMIT_RPM", "2")
    get_api_settings.cache_clear()
    limited = TestClient(create_app(), raise_server_exceptions=False)
    assert limited.get("/health").status_code == 200
    with patch(
        "src.api.routers.text.analyze_smart",
        return_value=([{"label": "positiv", "score": 0.5}], {"profile": "default"}),
    ):
        limited.post("/analyze", json={"texts": ["a"]})
        limited.post("/analyze", json={"texts": ["b"]})
        r = limited.post("/analyze", json={"texts": ["c"]})
    assert r.status_code == 429
    body = r.json()
    assert body["error_code"] == "rate_limit_exceeded"
