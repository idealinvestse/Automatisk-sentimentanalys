"""Unit tests for Fas 4 NiceGUI dashboard data helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.archive.nicegui_dashboard.services.demo_provider import load_demo_reports
from app.archive.nicegui_dashboard.services.fas4_data import (
    active_alerts,
    alert_dedup_key,
    fetch_agent_performance,
    fetch_semantic_search,
    list_agent_ids,
    local_agent_metrics,
    local_hot_topics_detailed,
    local_qa_from_report,
    local_semantic_search,
    reports_for_agent,
    reports_to_segments_list,
    resolve_call_id_from_hit,
)
from app.archive.nicegui_dashboard.services.nicegui_api_client import NiceGUIAPIClient


@pytest.fixture
def reports():
    load_demo_reports.cache_clear()
    return list(load_demo_reports())


class TestFas4DataLocal:
    def test_reports_to_segments_list(self, reports):
        segs = reports_to_segments_list(reports)
        assert len(segs) == 5
        assert all(isinstance(s, list) and s for s in segs)

    def test_list_agent_ids(self, reports):
        agents = list_agent_ids(reports)
        assert "Agent-Anna" in agents
        assert len(agents) == 5

    def test_local_agent_metrics(self, reports):
        metrics = local_agent_metrics("Agent-Anna", reports)
        assert metrics["call_count"] == 1
        assert "averages" in metrics

    def test_reports_for_agent(self, reports):
        anna = reports_for_agent(reports, "Agent-Anna")
        assert len(anna) == 1
        assert anna[0]["call_id"] == "CALL-001"

    def test_local_hot_topics(self, reports):
        topics = local_hot_topics_detailed(reports)
        assert isinstance(topics, list)
        if topics:
            assert topics[0].get("topic")

    def test_local_qa_from_report(self, reports):
        qa = local_qa_from_report(reports[0])
        assert qa.get("overall_qa_score") is not None

    def test_local_semantic_search_faktura(self, reports):
        hits = local_semantic_search("faktura fel", reports, top_k=3)
        assert hits
        assert hits[0]["score"] > 0
        assert hits[0].get("call_id")

    def test_resolve_call_id_from_hit(self, reports):
        hit = {"id": "0", "score": 0.5}
        assert resolve_call_id_from_hit(hit, reports) == "CALL-001"

    def test_alert_dedup_key(self):
        key = alert_dedup_key({"call_id": "C1", "rule_id": "low_empathy", "message": "x"})
        assert "C1" in key
        assert "low_empathy" in key

    def test_active_alerts_dismissed(self, reports):
        reports_with_alert = [
            {
                **reports[0],
                "results": {
                    **(reports[0].get("results") or {}),
                    "alerts": [
                        {
                            "rule_id": "test_rule",
                            "severity": "high",
                            "message": "Test alert",
                        }
                    ],
                },
            }
        ]
        alert = reports_with_alert[0]["results"]["alerts"][0]
        alert["call_id"] = reports_with_alert[0]["call_id"]
        key = alert_dedup_key(alert)
        active = active_alerts(reports_with_alert, dismissed_keys=[])
        assert len(active) == 1
        assert len(active_alerts(reports_with_alert, dismissed_keys=[key])) == 0


class TestFas4APIClient:
    @pytest.mark.asyncio
    async def test_get_agent_performance(self):
        client = NiceGUIAPIClient("http://test")
        payload = {"agent_id": "Agent-Anna", "metrics": {}, "cached": False}

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = payload

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=None)

        with patch_httpx(mock_http):
            result = await client.get_agent_performance(
                "Agent-Anna",
                [[{"text": "Hej", "speaker": "agent"}]],
            )
        assert result["agent_id"] == "Agent-Anna"
        path = mock_http.post.call_args.args[0]
        assert "/agent_performance/Agent-Anna" in path

    @pytest.mark.asyncio
    async def test_semantic_search(self):
        client = NiceGUIAPIClient("http://test")
        payload = {"query": "faktura", "hits": [], "meta": {}}

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = payload

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_response)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=None)

        with patch_httpx(mock_http):
            result = await client.semantic_search(
                "faktura",
                [[{"text": "faktura fel"}]],
                top_k=3,
            )
        assert result["query"] == "faktura"

    @pytest.mark.asyncio
    async def test_fetch_agent_performance_api_fallback(self, reports):
        client = AsyncMock()
        client.get_agent_performance = AsyncMock(
            return_value={"metrics": {"call_count": 2, "averages": {}}}
        )
        metrics, source = await fetch_agent_performance(client, "Agent-Anna", reports)
        assert source == "api"
        assert metrics["call_count"] == 2

    @pytest.mark.asyncio
    async def test_fetch_semantic_search_fallback_on_error(self, reports):
        client = AsyncMock()
        client.semantic_search = AsyncMock(side_effect=RuntimeError("down"))
        hits, source = await fetch_semantic_search(client, "faktura", reports)
        assert source == "local"
        assert isinstance(hits, list)


def patch_httpx(mock_http):
    from unittest.mock import patch

    return patch("httpx.AsyncClient", return_value=mock_http)
