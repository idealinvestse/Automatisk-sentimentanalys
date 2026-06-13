"""Tests for NiceGUI dashboard services (Fas 4).

Unit tests for api client, demo provider, transcription service, call detail helpers.
UI/E2E tests optional – NiceGUI runtime not required here.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.nicegui_dashboard.components.call_detail import find_report, _format_duration
from app.nicegui_dashboard.services.demo_provider import load_demo_reports, reports_to_table_rows
from app.nicegui_dashboard.services.nicegui_api_client import APIError, NiceGUIAPIClient
from app.nicegui_dashboard.services.transcription_service import (
    TranscriptionState,
    create_transcription_state,
)


class TestNiceGUIAPIClient:
    def test_from_env_defaults(self, monkeypatch):
        monkeypatch.delenv("SENTIMENT_API_BASE_URL", raising=False)
        monkeypatch.delenv("SENTIMENT_API_KEY", raising=False)
        client = NiceGUIAPIClient.from_env()
        assert client.base_url == "http://localhost:8000"
        assert client.api_key is None

    def test_headers_with_api_key(self):
        client = NiceGUIAPIClient("http://test", api_key="secret")
        headers = client._headers()
        assert headers["X-API-Key"] == "secret"

    @pytest.mark.asyncio
    async def test_health_ok(self):
        client = NiceGUIAPIClient("http://test")

        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_response)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_http):
            assert await client.health() is True

    @pytest.mark.asyncio
    async def test_analyze_pipeline_raises_on_error(self):
        client = NiceGUIAPIClient("http://test")

        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.json.return_value = {"detail": "validation error"}
        mock_response.text = "validation error"

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_http):
            with pytest.raises(APIError) as exc:
                await client.analyze_pipeline([{"text": "Hej"}])
            assert exc.value.status_code == 422

    @pytest.mark.asyncio
    async def test_analyze_pipeline_success(self):
        client = NiceGUIAPIClient("http://test")
        payload = {"sentiment_results": [], "results": {}}

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = payload

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_http):
            result = await client.analyze_pipeline([{"text": "Hej"}])
        assert result == payload


class TestDemoProvider:
    def test_load_demo_reports_returns_five(self):
        load_demo_reports.cache_clear()
        reports = list(load_demo_reports())
        assert len(reports) == 5

    def test_reports_to_table_rows(self):
        load_demo_reports.cache_clear()
        reports = list(load_demo_reports())
        rows = reports_to_table_rows(reports)
        assert len(rows) == 5
        assert "call_id" in rows[0]
        assert "sentiment" in rows[0]


class TestCallDetailHelpers:
    def test_find_report(self):
        reports = [{"call_id": "CALL-001"}, {"call_id": "CALL-002"}]
        assert find_report(reports, "CALL-002")["call_id"] == "CALL-002"
        assert find_report(reports, None) is None
        assert find_report(reports, "MISSING") is None

    def test_format_duration(self):
        assert _format_duration(125) == "2 min 5s"
        assert _format_duration(None) == "—"


class TestTranscriptionState:
    def test_save_and_load_queue(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "app.nicegui_dashboard.services.transcription_service.CACHE_DIR",
            tmp_path,
        )
        monkeypatch.setattr(
            "app.nicegui_dashboard.services.transcription_service.QUEUE_STATE_FILE",
            tmp_path / "transcription_queue.json",
        )
        state = TranscriptionState()
        state.queue = [Path("inputs/test.wav")]
        state.save()

        state2 = TranscriptionState()
        state2.load()
        assert len(state2.queue) == 1

    def test_add_log_truncates(self):
        state = TranscriptionState()
        for i in range(250):
            state.add_log("INFO", f"msg {i}")
        # Truncates to last 150 once buffer exceeds 200
        assert len(state.logs) < 250
        assert len(state.logs) <= 200

    def test_create_transcription_state(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "app.nicegui_dashboard.services.transcription_service.CACHE_DIR",
            tmp_path,
        )
        monkeypatch.setattr(
            "app.nicegui_dashboard.services.transcription_service.QUEUE_STATE_FILE",
            tmp_path / "transcription_queue.json",
        )
        state = create_transcription_state()
        assert isinstance(state, TranscriptionState)