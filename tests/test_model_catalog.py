"""Tests for OpenRouter model catalog fetch and load."""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.llm.model_catalog import fetch_openrouter_models_catalog, load_catalog


def _sample_api_response() -> dict:
    return {
        "data": [
            {
                "id": "mistralai/mistral-small-3.1-24b-instruct",
                "name": "Mistral Small",
                "description": "Fast model",
                "context_length": 128000,
                "pricing": {"prompt": "0.0000001", "completion": "0.0000003"},
                "architecture": {"modality": "text"},
                "top_provider": {"max_completion_tokens": 8192},
                "per_request_limits": None,
            }
        ]
    }


class TestFetchOpenRouterModelsCatalog:
    def test_fetch_saves_enriched_catalog(self, tmp_path: Path) -> None:
        out = tmp_path / "catalog.json"
        payload = json.dumps(_sample_api_response()).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.read.return_value = payload
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            catalog = fetch_openrouter_models_catalog(out, api_key="test-key")

        assert catalog["count"] == 1
        assert catalog["source"] == "https://openrouter.ai/api/v1/models"
        model = catalog["models"][0]
        assert model["id"] == "mistralai/mistral-small-3.1-24b-instruct"
        assert model["pricing"]["prompt_per_million_usd"] == 0.1
        assert model["pricing"]["completion_per_million_usd"] == 0.3
        assert out.is_file()
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["count"] == 1

    def test_fetch_without_api_key(self, tmp_path: Path) -> None:
        out = tmp_path / "catalog.json"
        payload = json.dumps({"data": []}).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.read.return_value = payload
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with (
            patch("urllib.request.urlopen", return_value=mock_resp) as mock_open,
            patch("src.llm.model_catalog.get_openrouter_api_key", return_value=None),
        ):
            catalog = fetch_openrouter_models_catalog(out)

        assert catalog["count"] == 0
        req = mock_open.call_args[0][0]
        assert "Authorization" not in req.headers

    def test_fetch_http_error_propagates(self, tmp_path: Path) -> None:
        err = urllib.error.HTTPError(
            url="https://openrouter.ai/api/v1/models",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=io.BytesIO(b"denied"),
        )
        with (
            patch("urllib.request.urlopen", side_effect=err),
            pytest.raises(urllib.error.HTTPError),
        ):
            fetch_openrouter_models_catalog(tmp_path / "catalog.json")

    def test_fetch_generic_error_propagates(self, tmp_path: Path) -> None:
        with (
            patch("urllib.request.urlopen", side_effect=OSError("network down")),
            pytest.raises(OSError, match="network down"),
        ):
            fetch_openrouter_models_catalog(tmp_path / "catalog.json")


class TestLoadCatalog:
    def test_load_missing_returns_none(self, tmp_path: Path) -> None:
        assert load_catalog(tmp_path / "missing.json") is None

    def test_load_valid_catalog(self, tmp_path: Path) -> None:
        path = tmp_path / "catalog.json"
        path.write_text(json.dumps({"count": 2, "models": []}), encoding="utf-8")
        loaded = load_catalog(path)
        assert loaded is not None
        assert loaded["count"] == 2

    def test_load_invalid_json_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("not-json", encoding="utf-8")
        assert load_catalog(path) is None
