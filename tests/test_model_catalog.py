"""Tests for OpenRouter model catalog fetch and load."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.llm.model_catalog import fetch_openrouter_models_catalog, load_catalog


def _sample_api_response() -> dict:
    return {
        "data": [
            {
                "id": "mistralai/mistral-small",
                "name": "Mistral Small",
                "description": "Fast model",
                "context_length": 32000,
                "pricing": {"prompt": "0.0000002", "completion": "0.0000006"},
                "architecture": {"modality": "text"},
                "top_provider": {"max_completion_tokens": 8192},
                "per_request_limits": None,
            }
        ]
    }


def test_fetch_openrouter_models_catalog_saves_enriched_json(tmp_path: Path) -> None:
    out = tmp_path / "catalog.json"
    payload = json.dumps(_sample_api_response()).encode("utf-8")

    def fake_urlopen(req, timeout=60):
        resp = MagicMock()
        resp.read.return_value = payload
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        catalog = fetch_openrouter_models_catalog(output_path=out, api_key="test-key")

    assert catalog["count"] == 1
    assert catalog["models"][0]["id"] == "mistralai/mistral-small"
    assert catalog["models"][0]["pricing"]["prompt_per_million_usd"] == 0.2
    assert out.is_file()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["count"] == 1


def test_fetch_openrouter_models_catalog_without_api_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    out = tmp_path / "catalog.json"
    payload = json.dumps({"data": []}).encode("utf-8")

    def fake_urlopen(req, timeout=60):
        assert "Authorization" not in req.headers
        resp = MagicMock()
        resp.read.return_value = payload
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        catalog = fetch_openrouter_models_catalog(output_path=out, api_key=None)

    assert catalog["count"] == 0


def test_fetch_openrouter_models_catalog_http_error(tmp_path: Path) -> None:
    import urllib.error

    def fake_urlopen(req, timeout=60):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", hdrs=None, fp=BytesIO())

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        with pytest.raises(urllib.error.HTTPError):
            fetch_openrouter_models_catalog(output_path=tmp_path / "c.json", api_key="bad")


def test_fetch_openrouter_models_catalog_generic_failure(tmp_path: Path) -> None:
    with patch("urllib.request.urlopen", side_effect=RuntimeError("network down")):
        with pytest.raises(RuntimeError, match="network down"):
            fetch_openrouter_models_catalog(output_path=tmp_path / "c.json")


def test_load_catalog_missing_returns_none(tmp_path: Path) -> None:
    assert load_catalog(tmp_path / "missing.json") is None


def test_load_catalog_valid(tmp_path: Path) -> None:
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"count": 2, "models": []}), encoding="utf-8")
    loaded = load_catalog(path)
    assert loaded is not None
    assert loaded["count"] == 2


def test_load_catalog_invalid_json_returns_none(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("not-json", encoding="utf-8")
    assert load_catalog(path) is None
