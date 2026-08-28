"""Fail if checked-in webui/openapi.json drifts from the live app OpenAPI schema."""

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


def test_webui_openapi_matches_live_contract() -> None:
    live = TestClient(create_app()).get("/openapi.json").json()
    checked = json.loads(WEBUI_OPENAPI.read_text(encoding="utf-8"))

    live_ver = live.get("info", {}).get("version")
    checked_ver = checked.get("info", {}).get("version")
    assert live_ver == checked_ver, (
        f"Version drift between live ({live_ver}) and webui/openapi.json ({checked_ver})"
    )

    live_paths = set(live.get("paths") or {})
    checked_paths = set(checked.get("paths") or {})
    missing = sorted(live_paths - checked_paths)
    extra = sorted(checked_paths - live_paths)
    assert not missing and not extra, (
        f"webui/openapi.json path drift: missing={missing}, extra={extra}"
    )

    live_schemas = (live.get("components") or {}).get("schemas") or {}
    checked_schemas = (checked.get("components") or {}).get("schemas") or {}
    missing_schemas = sorted(set(live_schemas) - set(checked_schemas))
    extra_schemas = sorted(set(checked_schemas) - set(live_schemas))
    changed_schemas = sorted(
        name
        for name in set(live_schemas) & set(checked_schemas)
        if live_schemas[name] != checked_schemas[name]
    )
    assert not missing_schemas and not extra_schemas and not changed_schemas, (
        "webui/openapi.json schema drift: "
        f"missing={missing_schemas}, extra={extra_schemas}, changed={changed_schemas}"
    )

    changed_operations = sorted(
        path
        for path in live_paths & checked_paths
        if live["paths"][path] != checked["paths"][path]
    )
    assert not changed_operations, (
        f"webui/openapi.json operation drift: changed={changed_operations}"
    )
