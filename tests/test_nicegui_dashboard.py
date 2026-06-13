"""Tests for NiceGUI dashboard services (Fas 4).

Unit tests for api client, demo provider, transcription service, call detail helpers.
UI/E2E tests optional – NiceGUI runtime not required here.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.nicegui_dashboard.components.call_detail import (
    _build_insights_markdown,
    find_report,
    _format_duration,
)
from app.nicegui_dashboard.services.nicegui_api_client import JOB_HEADER
from app.nicegui_dashboard.services.calls_filter import (
    format_search_hit_label,
    paginate_items,
    search_table_reports,
)
from app.nicegui_dashboard.services.qa_display import qa_score_css_class, qa_score_tier
from app.nicegui_dashboard.services.chart_data import (
    build_agent_trends_figure,
    build_escalation_figure,
    build_hot_topics_figure,
    build_trajectory_figure,
    call_id_from_plotly_click,
    extract_agent_trend_rows,
    extract_trajectory_points,
    list_call_options,
)
from app.nicegui_dashboard.services.demo_provider import load_demo_reports, reports_to_table_rows
from app.nicegui_dashboard.services.transcript_virtualizer import (
    VIRTUALIZE_THRESHOLD,
    compute_visible_range,
    filter_segments_with_index,
    highlight_search_text,
    make_synthetic_segments,
    should_virtualize,
    window_around_index,
)
from app.nicegui_dashboard.services.transcription_ws_client import (
    WS_CONNECTED,
    WS_DISCONNECTED,
    WS_RECONNECTING,
    TranscriptionWSListener,
)
from app.nicegui_dashboard.state import DashboardState
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

    @pytest.mark.asyncio
    async def test_transcribe_includes_job_header(self):
        client = NiceGUIAPIClient("http://test")
        client.set_job_id("job-abc")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"transcript": {"segments": []}}

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_http):
            await client.transcribe("/audio/test.wav")

        call_kwargs = mock_http.post.call_args
        assert call_kwargs.kwargs["headers"][JOB_HEADER] == "job-abc"

    @pytest.mark.asyncio
    async def test_wait_for_health_retries(self):
        client = NiceGUIAPIClient("http://test")
        with patch.object(client, "health", new_callable=AsyncMock) as mock_health:
            mock_health.side_effect = [False, False, True]
            ok = await client.wait_for_health(attempts=3, interval=0.01)
        assert ok is True
        assert mock_health.call_count == 3


class TestCallsTableFilter:
    def test_search_by_call_id(self):
        reports = [
            {"call_id": "CALL-001", "title": "A", "meta": {"agent": "Anna"}, "segments": []},
            {"call_id": "CALL-002", "title": "B", "meta": {"agent": "Bengt"}, "segments": []},
        ]
        found = search_table_reports(reports, "call-002")
        assert len(found) == 1
        assert found[0]["call_id"] == "CALL-002"

    def test_search_in_segment_text(self):
        reports = [
            {
                "call_id": "CALL-001",
                "title": "X",
                "meta": {"agent": "Anna"},
                "segments": [{"text": "internationellt samtal till Tyskland"}],
            },
            {
                "call_id": "CALL-002",
                "title": "Y",
                "meta": {"agent": "Bengt"},
                "segments": [{"text": "fakturafråga"}],
            },
        ]
        found = search_table_reports(reports, "tyskland")
        assert len(found) == 1
        assert found[0]["call_id"] == "CALL-001"

    def test_paginate_second_page(self):
        items = list(range(25))
        page_slice, total_pages, total = paginate_items(items, page=2, page_size=20)
        assert total == 25
        assert total_pages == 2
        assert page_slice == list(range(20, 25))

    def test_paginate_empty(self):
        page_slice, total_pages, total = paginate_items([], page=1, page_size=20)
        assert page_slice == []
        assert total_pages == 1
        assert total == 0

    def test_dashboard_state_table_defaults(self):
        state = DashboardState(reports=[])
        assert state.table_page == 1
        assert state.table_page_size == 20
        assert state.table_search == ""

    def test_paginated_table_rows_slice(self):
        load_demo_reports.cache_clear()
        reports = list(load_demo_reports())
        searched = search_table_reports(reports, "")
        page_rows, total_pages, total = paginate_items(searched, page=1, page_size=2)
        table_rows = reports_to_table_rows(page_rows)
        assert total == 5
        assert total_pages == 3
        assert len(table_rows) == 2


class TestTranscriptVirtualizer:
    def test_should_virtualize_threshold(self):
        assert should_virtualize(VIRTUALIZE_THRESHOLD - 1) is False
        assert should_virtualize(VIRTUALIZE_THRESHOLD) is True
        assert should_virtualize(150) is True

    def test_compute_visible_range(self):
        start, end = compute_visible_range(scroll_top=500, container_height=300, total_items=200)
        assert start >= 0
        assert end > start
        assert end <= 200

    def test_window_around_index(self):
        start, end = window_around_index(50, 120, window_size=20)
        assert start <= 50 < end
        assert end - start <= 20

    def test_filter_segments_with_index(self):
        enriched = [
            {"text": "faktura fel", "speaker": "Kund"},
            {"text": "teknisk support", "speaker": "Agent"},
        ]
        found = filter_segments_with_index(enriched, "faktura")
        assert len(found) == 1
        assert found[0][0] == 0

    def test_make_synthetic_segments_count(self):
        segs = make_synthetic_segments(120)
        assert len(segs) == 120
        assert segs[0]["speaker"] == "Agent"


class TestChartData:
    def test_extract_trajectory_points_from_segments(self):
        report = {
            "call_id": "CALL-001",
            "segments": [
                {"start": 0, "text": "Hej tack för hjälpen", "speaker": "Agent"},
                {"start": 10, "text": "Jag är arg och frustrerad", "speaker": "Kund"},
            ],
            "sentiment_results": [
                {"label": "neutral", "score": 0.5},
                {"label": "neutral", "score": 0.5},
            ],
        }
        points = extract_trajectory_points(report)
        assert len(points) == 2
        assert points[0]["y"] > 0
        assert points[1]["y"] < 0

    def test_extract_agent_trend_rows(self):
        load_demo_reports.cache_clear()
        reports = list(load_demo_reports())
        rows = extract_agent_trend_rows(reports)
        assert len(rows) == 5
        assert "empathy" in rows[0]
        assert "qa" in rows[0]
        assert "call_id" in rows[0]

    def test_build_trajectory_figure_has_trace(self):
        load_demo_reports.cache_clear()
        report = list(load_demo_reports())[0]
        fig = build_trajectory_figure(report)
        assert len(fig.data) >= 1

    def test_build_agent_trends_figure_dual_axis(self):
        load_demo_reports.cache_clear()
        rows = extract_agent_trend_rows(list(load_demo_reports()))
        fig = build_agent_trends_figure(rows)
        assert len(fig.data) == 2

    def test_build_hot_topics_figure_fallback_categories(self):
        load_demo_reports.cache_clear()
        reports = list(load_demo_reports())
        fig = build_hot_topics_figure(reports)
        assert len(fig.data) == 1

    def test_build_escalation_figure(self):
        rows = [{"call_id": "C1", "title": "T", "escalation": 2}]
        fig = build_escalation_figure(rows)
        assert fig.data[0].y == (2,)

    def test_list_call_options(self):
        reports = [{"call_id": "CALL-001", "title": "Test"}]
        opts = list_call_options(reports)
        assert opts[0]["value"] == "CALL-001"

    def test_call_id_from_plotly_click(self):
        event = {"points": [{"customdata": "CALL-002"}]}
        assert call_id_from_plotly_click(event) == "CALL-002"


class TestWebSocketReconnect:
    def test_status_constants(self):
        assert WS_CONNECTED == "connected"
        assert WS_RECONNECTING == "reconnecting"
        assert WS_DISCONNECTED == "disconnected"

    def test_set_status_notifies_handler(self):
        client = NiceGUIAPIClient("http://test")
        statuses: list[str] = []
        listener = TranscriptionWSListener(
            client,
            on_event=lambda _e: None,
            on_status_change=statuses.append,
        )
        listener._set_status(WS_RECONNECTING)
        listener._set_status(WS_CONNECTED)
        assert statuses == [WS_RECONNECTING, WS_CONNECTED]

    def test_needs_polling_fallback(self):
        state = TranscriptionState()
        state.status["use_api"] = True
        state.status["is_running"] = True
        state.ws_status = "disconnected"
        assert state.needs_polling_fallback() is True
        state.ws_status = "connected"
        assert state.needs_polling_fallback() is False

    @pytest.mark.asyncio
    async def test_request_ws_reconnect_without_api(self):
        state = TranscriptionState()
        state.status["use_api"] = False
        await state.request_ws_reconnect()
        assert state.ws_status == "disconnected"

    @pytest.mark.asyncio
    async def test_reconnect_now_resets_attempt(self):
        client = NiceGUIAPIClient("http://test")
        listener = TranscriptionWSListener(client, on_event=lambda _e: None)
        listener._attempt = 5
        listener._stop.set()
        with patch.object(listener, "_listen_loop", new_callable=AsyncMock):
            await listener.reconnect_now("job-1")
        assert listener._attempt == 0
        assert listener.job_id == "job-1"


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
        assert "qa_class" in rows[0]


class TestFas62Helpers:
    def test_highlight_search_text_case_insensitive(self):
        out = highlight_search_text("Hej Faktura och faktura", "faktura")
        assert out.count("search-hit") == 2
        assert "Faktura" in out

    def test_highlight_search_text_escapes_html(self):
        out = highlight_search_text("<script>alert(1)</script>", "script")
        assert "<script>" not in out
        assert "search-hit" in out

    def test_highlight_search_text_empty_query(self):
        assert highlight_search_text("plain text", "") == "plain text"

    def test_qa_score_tiers(self):
        assert qa_score_tier(85) == "high"
        assert qa_score_tier(70) == "mid"
        assert qa_score_tier(45) == "low"
        assert qa_score_tier(None) == "none"
        assert qa_score_css_class(85) == "text-positive"
        assert qa_score_css_class("—") == "text-grey"

    def test_format_search_hit_label(self):
        assert "1 träff" in format_search_hit_label(1, "faktura")
        assert "3 träffar" in format_search_hit_label(3, "faktura")

    def test_plotly_figures_have_hovertemplate(self):
        load_demo_reports.cache_clear()
        reports = list(load_demo_reports())
        report = reports[0]
        trend_rows = extract_agent_trend_rows(reports)
        assert build_trajectory_figure(report).data[0].hovertemplate
        assert build_agent_trends_figure(trend_rows).data[0].hovertemplate
        assert build_escalation_figure(trend_rows).data[0].hovertemplate


class TestCallDetailHelpers:
    def test_find_report(self):
        reports = [{"call_id": "CALL-001"}, {"call_id": "CALL-002"}]
        assert find_report(reports, "CALL-002")["call_id"] == "CALL-002"
        assert find_report(reports, None) is None
        assert find_report(reports, "MISSING") is None

    def test_format_duration(self):
        assert _format_duration(125) == "2 min 5s"
        assert _format_duration(None) == "—"

    def test_build_insights_markdown_with_qa(self):
        report = {
            "results": {"qa": {"overall_qa_score": 88}},
            "llm": {},
        }
        md = _build_insights_markdown(report)
        assert "88" in md


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

    def test_start_batch_empty_queue(self, tmp_path):
        state = TranscriptionState()
        state.pending_folder = str(tmp_path)
        assert state.start_batch() is False

    def test_pause_resume_stop(self):
        state = TranscriptionState()
        state.request_pause()
        assert state.pause is True
        state.request_resume()
        assert state.pause is False
        state.request_stop()
        assert state.stop is True

    def test_listener_notifies_on_log(self):
        state = TranscriptionState()
        events: list[str] = []
        state.add_listener(lambda t, _p: events.append(t))
        state.add_log("INFO", "hej")
        assert "log" in events

    def test_set_ws_status(self):
        state = TranscriptionState()
        state._set_ws_status("reconnecting")
        assert state.ws_status == "reconnecting"
        assert state.ws_connected is False
        state._set_ws_status("connected")
        assert state.ws_connected is True