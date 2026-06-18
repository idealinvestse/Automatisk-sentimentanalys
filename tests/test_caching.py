"""Tests for Fas 4.5 AggregateCache and precompute helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest  # noqa: F401

from src.caching import (
    AggregateCache,
    precompute_and_cache,
    precompute_agent_aggregates,
    precompute_hot_topics,
)


@pytest.fixture
def cache(tmp_path):
    return AggregateCache(cache_dir=str(tmp_path / "aggregates"))


class TestAggregateCache:
    def test_set_and_get_file_cache(self, cache):
        cache.set("test-key", {"call_count": 5})
        result = cache.get("test-key")
        assert result is not None
        assert result["call_count"] == 5
        assert "computed_at" in result
        assert "ttl" in result

    def test_cache_miss(self, cache):
        assert cache.get("nonexistent") is None

    def test_precompute_and_cache_hit_flag(self, cache):
        calls = {"n": 0}

        def compute():
            calls["n"] += 1
            return {"value": 42}

        r1 = precompute_and_cache(cache, "k1", compute)
        r2 = precompute_and_cache(cache, "k1", compute)
        assert r1["cache_hit"] is False
        assert r2["cache_hit"] is True
        assert r1["value"] == 42
        assert calls["n"] == 1

    def test_invalidate_removes_matching_entries(self, cache):
        cache.set("key-a", {"marker": "agent:Agent-1:7d:2", "call_count": 2})
        cache.set("key-b", {"marker": "agent:Agent-2:7d:2", "call_count": 1})
        cache.invalidate("agent:Agent-1")
        assert cache.get("key-a") is None
        assert cache.get("key-b") is not None

    def test_is_valid_rejects_expired_entry(self, cache):
        from datetime import datetime, timedelta

        from src.caching import AggregateCache

        old = (datetime.now() - timedelta(days=2)).isoformat()
        assert AggregateCache._is_valid(cache, {"computed_at": old, "ttl": 3600}) is False


class TestPrecomputeHelpers:
    def test_precompute_agent_aggregates_without_cache(self):
        from src.agent_performance import compute_call_agent_performance

        segments = [
            {"start": 0, "end": 2, "text": "Hej välkommen.", "speaker": "A"},
            {"start": 2, "end": 5, "text": "Faktura fel.", "speaker": "C"},
        ]
        perf = compute_call_agent_performance(
            segments, role_map={"A": "agent", "C": "customer"}
        )
        reports = [{"agent_performance": perf.model_dump()}]
        result = precompute_agent_aggregates(reports, agent_id="Agent-1")
        assert result["call_count"] == 1
        assert "averages" in result

    def test_precompute_hot_topics_without_cache(self, monkeypatch):
        monkeypatch.setattr(
            "src.insights_aggregator.InsightsAggregator.aggregate",
            lambda self, reports: MagicMock(model_dump=lambda: {"hot_topics": [], "meta": {}}),
        )
        result = precompute_hot_topics([{"segments": [{"text": "faktura"}]}])
        assert "hot_topics" in result