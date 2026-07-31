"""Tests for multi-provider secrets, catalog, and router."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.llm.model_catalog import fetch_provider_models_catalog
from src.llm.multi_provider_router import MultiProviderRouter, RateLimitTracker, RouterProfile
from src.llm.openai_compat_client import OpenAICompatClient
from src.llm.provider_secrets import get_provider_api_key, list_configured_providers


def test_get_provider_api_key_from_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key_file = tmp_path / "mistral.key"
    key_file.write_text("test-mistral-key-123\n", encoding="utf-8")
    cfg = {
        "providers": {
            "mistral": {
                "env_keys": ["MISTRAL_API_KEY_TEST"],
                "key_files": [str(key_file)],
            }
        }
    }
    monkeypatch.delenv("MISTRAL_API_KEY_TEST", raising=False)
    assert get_provider_api_key("mistral", config=cfg) == "test-mistral-key-123"


def test_fetch_provider_catalog_normalizes(tmp_path: Path) -> None:
    out = tmp_path / "nvidia.json"
    payload = json.dumps(
        {
            "data": [
                {"id": "meta/llama-3.1-8b-instruct", "object": "model", "owned_by": "meta"},
                {"id": "paid-model", "pricing": {"prompt": "0.000001", "completion": "0.000002"}},
            ]
        }
    ).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = payload
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)

    cfg = {
        "providers": {
            "nvidia": {
                "base_url": "https://example.test/v1",
                "models_path": "/models",
                "curated_free": ["meta/llama-3.1-8b-instruct"],
                "env_keys": [],
                "key_files": [],
            }
        },
        "catalog": {"dir": str(tmp_path)},
    }
    with (
        patch("src.llm.model_catalog._http_get_json", return_value=json.loads(payload)),
        patch("src.llm.model_catalog.get_provider_api_key", return_value="k"),
    ):
        cat = fetch_provider_models_catalog("nvidia", output_path=out, api_key="k", config=cfg)

    assert cat["count"] == 2
    free = {m["id"] for m in cat["models"] if m.get("is_free")}
    assert "meta/llama-3.1-8b-instruct" in free
    assert out.is_file()


def test_rate_limit_tracker_cooldown(tmp_path: Path) -> None:
    tracker = RateLimitTracker(tmp_path / "rate.json")
    assert tracker.is_available("nvidia", rpm=2)
    tracker.record_success("nvidia")
    tracker.record_success("nvidia")
    assert tracker.is_available("nvidia", rpm=2) is False
    tracker.record_rate_limit("cerebras", cooldown_seconds=30)
    assert tracker.is_available("cerebras") is False


def test_router_selects_free_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cat_dir = tmp_path / "cats"
    cat_dir.mkdir()
    (cat_dir / "nvidia.json").write_text(
        json.dumps(
            {
                "models": [
                    {"id": "meta/llama-3.1-8b-instruct", "is_free": True},
                    {"id": "big-paid", "is_free": False},
                ]
            }
        ),
        encoding="utf-8",
    )
    cfg = {
        "providers": {
            "nvidia": {
                "enabled": True,
                "base_url": "https://example.test/v1",
                "env_keys": ["NVIDIA_API_KEY_TEST"],
                "key_files": [],
                "curated_free": ["meta/llama-3.1-8b-instruct"],
                "default_rpm": 40,
            },
            "mistral": {"enabled": True, "env_keys": ["NOPE"], "key_files": [], "base_url": "x"},
        },
        "profiles": {
            "free_sequential": {
                "free_only": True,
                "provider_order": ["nvidia", "mistral"],
                "max_provider_attempts": 2,
                "cooldown_seconds_on_429": 60,
            }
        },
        "catalog": {
            "dir": str(cat_dir),
            "rate_state_file": str(tmp_path / "rate.json"),
        },
    }
    monkeypatch.setenv("NVIDIA_API_KEY_TEST", "nv-test-key")
    with patch("src.llm.multi_provider_router.load_provider_catalog") as load_cat:
        load_cat.return_value = json.loads((cat_dir / "nvidia.json").read_text(encoding="utf-8"))
        router = MultiProviderRouter(profile=RouterProfile.FREE_SEQUENTIAL, config=cfg)
        choice = router.select_route()
    assert choice.provider == "nvidia"
    assert choice.model == "meta/llama-3.1-8b-instruct"
    assert choice.profile == "free_sequential"


def test_openai_compat_client_missing_key() -> None:
    from src.core.errors import LLMError

    client = OpenAICompatClient(
        provider="mistral",
        api_key=None,
        base_url="https://api.mistral.ai/v1",
        default_model="mistral-small-latest",
    )
    # force empty
    client.api_key = None
    with pytest.raises(LLMError):
        client.chat_completion([{"role": "user", "content": "hej"}])


def test_list_configured_providers_smoke() -> None:
    # Should not crash; returns dict
    status = list_configured_providers()
    assert isinstance(status, dict)
