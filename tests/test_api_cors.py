"""CORS allowlist / deny tests for the FastAPI app."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.settings import get_api_settings

ALLOWED = "https://dashboard.example.com"
DENIED = "https://evil.example"


@pytest.fixture
def cors_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.delenv("SENTIMENT_API_KEY", raising=False)
    monkeypatch.setenv("API_CORS_ORIGINS", ALLOWED)
    get_api_settings.cache_clear()
    return TestClient(create_app())


def test_cors_allows_configured_origin(cors_client: TestClient) -> None:
    response = cors_client.options(
        "/analyze",
        headers={
            "Origin": ALLOWED,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code in (200, 204)
    assert response.headers.get("access-control-allow-origin") == ALLOWED


def test_cors_denies_unknown_origin(cors_client: TestClient) -> None:
    response = cors_client.options(
        "/analyze",
        headers={
            "Origin": DENIED,
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.headers.get("access-control-allow-origin") != DENIED


def test_cors_reflects_origin_on_simple_get(cors_client: TestClient) -> None:
    allowed = cors_client.get("/health", headers={"Origin": ALLOWED})
    assert allowed.headers.get("access-control-allow-origin") == ALLOWED
    denied = cors_client.get("/health", headers={"Origin": DENIED})
    assert denied.headers.get("access-control-allow-origin") != DENIED
