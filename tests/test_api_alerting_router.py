"""Tests for /alerting API router endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api import app as default_app
from src.api.settings import get_api_settings


@pytest.fixture(autouse=True)
def _clear_api_settings_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SENTIMENT_API_KEY", raising=False)
    get_api_settings.cache_clear()


client = TestClient(default_app, raise_server_exceptions=False)


def test_get_alerting_status() -> None:
    r = client.get("/alerting/status")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "webhook" in body
    assert "Circuit breaker" in body["note"]


def test_reset_circuit_breaker() -> None:
    default_app.state.alert_engine._webhook_disabled = True
    default_app.state.alert_engine._consecutive_failures = 5

    r = client.post("/alerting/reset-circuit-breaker")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "reset" in body["message"].lower()
    assert default_app.state.alert_engine._webhook_disabled is False


def test_alerting_status_internal_error_returns_500() -> None:
    engine = MagicMock()
    engine.get_webhook_status.side_effect = RuntimeError("boom")

    with patch.object(default_app.state, "alert_engine", engine):
        r = client.get("/alerting/status")
    assert r.status_code == 500
