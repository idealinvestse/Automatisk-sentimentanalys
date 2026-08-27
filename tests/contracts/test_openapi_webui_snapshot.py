"""Fail if checked-in webui/openapi.json is missing live app paths."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.settings import get_api_settings

WEBUI_OPENAPI = Path(__file__).resolve().parents[2] / "webui" / "openapi.json"


@pytest.fixture(autouse=True)
def _clear_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SENTIMENT_API_KEY", raising=False)
    get_api_settings.cache_clear()


def test_webui_openapi_contains_live_paths() -> None:
    live = TestClient(create_app()).get("/openapi.json").json()
    checked = json.loads(WEBUI_OPENAPI.read_text(encoding="utf-8"))
    live_paths = set(live.get("paths") or {})
    checked_paths = set(checked.get("paths") or {})
    missing = sorted(live_paths - checked_paths)
    assert missing == [], f"webui/openapi.json missing live paths: {missing}"
