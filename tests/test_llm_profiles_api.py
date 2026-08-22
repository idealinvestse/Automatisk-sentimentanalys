"""HTTP tests for GET /llm/* routers."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.settings import get_api_settings


@pytest.fixture(autouse=True)
def _clear_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SENTIMENT_API_KEY", raising=False)
    get_api_settings.cache_clear()


def test_llm_profiles_list_and_providers() -> None:
    client = TestClient(create_app())
    fake_snap = {
        "profiles": [
            {
                "id": "cost_saver",
                "label": "Kostnadssparare",
                "description": "cheap",
                "use_when": "batch",
                "icon": "wallet",
                "cost_priority": 0.9,
                "quality_priority": 0.25,
                "recommended": {"model_id": "x", "provider": "openrouter"},
                "selectable": True,
            }
        ]
    }
    with (
        patch("src.api.routers.llm_profiles.list_analysis_profiles", return_value=fake_snap),
        patch(
            "src.api.routers.llm_profiles.list_configured_providers",
            return_value={"openrouter": True, "groq": False},
        ),
    ):
        listed = client.get("/llm/analysis-profiles")
        providers = client.get("/llm/providers")
    assert listed.status_code == 200
    body = listed.json()
    assert body["cached"] is False
    assert body["providers_configured"]["openrouter"] is True
    assert "sk-" not in str(body)
    assert any(item["id"] == "cost_saver" for item in body["menu"])
    assert providers.status_code == 200
    assert providers.json()["providers"]["groq"] is False
    assert all(isinstance(v, bool) for v in providers.json()["providers"].values())


def test_llm_profile_unknown_404() -> None:
    client = TestClient(create_app())
    r = client.get("/llm/analysis-profiles/not-a-perspective")
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert detail["error"] == "unknown_perspective"
    assert "cost_saver" in detail["available"]


def test_llm_profiles_cached_snapshot() -> None:
    client = TestClient(create_app())
    snap = {
        "profiles": [
            {
                "id": "cost_saver",
                "label": "Kostnadssparare",
                "description": "cheap",
                "use_when": "batch",
                "icon": "wallet",
                "recommended": {"model_id": "x", "provider": "openrouter"},
                "selectable": True,
            }
        ]
    }
    with (
        patch("src.api.routers.llm_profiles.load_profiles_snapshot", return_value=snap),
        patch(
            "src.api.routers.llm_profiles.list_configured_providers",
            return_value={"openrouter": True},
        ),
    ):
        listed = client.get("/llm/analysis-profiles?refresh=false")
    assert listed.status_code == 200
    body = listed.json()
    assert body["cached"] is True
    assert any(item["id"] == "cost_saver" for item in body["menu"])


def test_llm_profile_detail_happy() -> None:
    client = TestClient(create_app())

    class _Rec:
        def to_public(self) -> dict:
            return {"id": "cost_saver", "recommended": {"model_id": "x"}}

    with (
        patch("src.api.routers.llm_profiles.recommend_for_perspective", return_value=_Rec()),
        patch(
            "src.api.routers.llm_profiles.list_configured_providers",
            return_value={"openrouter": True},
        ),
    ):
        detail = client.get("/llm/analysis-profiles/cost_saver")
    assert detail.status_code == 200
    assert detail.json()["id"] == "cost_saver"
    assert detail.json()["providers_configured"]["openrouter"] is True


def test_llm_profiles_require_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENTIMENT_API_KEY", "secret")
    get_api_settings.cache_clear()
    client = TestClient(create_app())
    denied = client.get("/llm/providers")
    assert denied.status_code == 401
    assert denied.json()["error_code"] == "unauthorized"
    ok = client.get("/llm/providers", headers={"X-API-Key": "secret"})
    assert ok.status_code == 200
