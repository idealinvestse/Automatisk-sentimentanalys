"""Tests for alerting API router endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from src.api import app as default_app
from src.api.settings import get_api_settings

client = TestClient(default_app, raise_server_exceptions=False)


def test_alerting_status_returns_webhook_health() -> None:
    engine = MagicMock()
    engine.get_webhook_status.return_value = {
        "enabled": True,
        "url_configured": True,
        "circuit_breaker_open": False,
        "consecutive_failures": 0,
    }
    default_app.state.alert_engine = engine

    r = client.get("/alerting/status")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["webhook"]["enabled"] is True
    assert "Circuit breaker" in body["note"]
    engine.get_webhook_status.assert_called_once()


def test_reset_circuit_breaker_endpoint() -> None:
    engine = MagicMock()
    engine.get_webhook_status.return_value = {"circuit_breaker_open": False}
    default_app.state.alert_engine = engine

    r = client.post("/alerting/reset-circuit-breaker")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "reset" in body["message"].lower()
    engine.reset_circuit_breaker.assert_called_once()


def test_alerting_status_internal_error_sanitized() -> None:
    engine = MagicMock()
    engine.get_webhook_status.side_effect = RuntimeError("db down")
    default_app.state.alert_engine = engine

    r = client.get("/alerting/status")
    assert r.status_code == 500
    assert r.json()["detail"] == "An internal error occurred. Please try again later."


def test_alerting_status_requires_no_api_key(monkeypatch) -> None:
    monkeypatch.setenv("SENTIMENT_API_KEY", "secret")
    get_api_settings.cache_clear()
    authed = TestClient(default_app, raise_server_exceptions=False)
    r = authed.get("/alerting/status")
    assert r.status_code == 401
    get_api_settings.cache_clear()
