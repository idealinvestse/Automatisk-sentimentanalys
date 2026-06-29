"""Tests for Fas 4.5 AggregateCache and precompute helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest  # noqa: F401

from src.caching import (
    AggregateCache,
    precompute_agent_aggregates,
    precompute_and_cache,
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
        perf = compute_call_agent_performance(segments, role_map={"A": "agent", "C": "customer"})
        reports = [{"agent_performance": perf.model_dump()}]
        result = precompute_agent_aggregates(reports, agent_id="Agent-1")
        assert result["call_count"] == 1
        assert "averages" in result

    def test_precompute_agent_aggregates_with_cache_hit(self, cache):
        from src.agent_performance import compute_call_agent_performance

        segments = [
            {"start": 0, "end": 2, "text": "Hej välkommen.", "speaker": "A"},
            {"start": 2, "end": 5, "text": "Faktura fel.", "speaker": "C"},
        ]
        perf = compute_call_agent_performance(segments, role_map={"A": "agent", "C": "customer"})
        reports = [{"agent_performance": perf.model_dump()}]
        r1 = precompute_agent_aggregates(reports, cache=cache, agent_id="Agent-1")
        r2 = precompute_agent_aggregates(reports, cache=cache, agent_id="Agent-1")
        assert r1.get("cache_hit") is False
        assert r2.get("cache_hit") is True

    def test_precompute_hot_topics_with_cache(self, cache, monkeypatch):
        monkeypatch.setattr(
            "src.insights_aggregator.InsightsAggregator.aggregate",
            lambda self, reports: MagicMock(model_dump=lambda: {"topics": []}),
        )
        r1 = precompute_hot_topics([], cache=cache)
        r2 = precompute_hot_topics([], cache=cache)
        assert r1.get("cache_hit") is False
        assert r2.get("cache_hit") is True

    def test_precompute_hot_topics_without_cache(self, monkeypatch):
        monkeypatch.setattr(
            "src.insights_aggregator.InsightsAggregator.aggregate",
            lambda self, reports: MagicMock(model_dump=lambda: {"hot_topics": [], "meta": {}}),
        )
        result = precompute_hot_topics([{"segments": [{"text": "faktura"}]}])
        assert "hot_topics" in result


class TestAggregateCacheEdgeCases:
    def test_get_returns_none_for_corrupt_file(self, cache):
        path = cache._file_path("bad-key")
        path.write_text("{not-json", encoding="utf-8")
        assert cache.get("bad-key") is None

    def test_make_key_is_deterministic(self, cache):
        k1 = cache._make_key("prefix", "a", "b")
        k2 = cache._make_key("prefix", "a", "b")
        assert k1 == k2
        assert len(k1) == 16

    def test_is_valid_without_timestamps(self, cache):
        assert cache._is_valid({"value": 1}) is True

    def test_is_valid_rejects_bad_timestamp(self, cache):
        assert cache._is_valid({"computed_at": "not-a-date", "ttl": 3600}) is True

    def test_redis_fallback_when_unavailable(self, tmp_path, monkeypatch):
        fake_redis = MagicMock()
        fake_redis.from_url.side_effect = ConnectionError("redis down")
        monkeypatch.setattr("src.caching.REDIS_AVAILABLE", True)
        monkeypatch.setattr("src.caching.redis", fake_redis)
        cache = AggregateCache(use_redis=True, cache_dir=str(tmp_path / "agg"))
        assert cache.use_redis is False
        cache.set("k", {"x": 1})
        assert cache.get("k") is not None

    def test_redis_get_hit(self, tmp_path, monkeypatch):
        import json

        fake_client = MagicMock()
        fake_client.ping.return_value = True
        payload = json.dumps(
            {"value": 99, "computed_at": "2099-01-01T00:00:00", "ttl": 999999}
        )
        fake_client.get.return_value = payload
        fake_redis = MagicMock()
        fake_redis.from_url.return_value = fake_client
        monkeypatch.setattr("src.caching.REDIS_AVAILABLE", True)
        monkeypatch.setattr("src.caching.redis", fake_redis)
        cache = AggregateCache(use_redis=True, cache_dir=str(tmp_path / "agg"))
        result = cache.get("redis-key")
        assert result is not None
        assert result["value"] == 99

    def test_redis_set_failure_falls_back_to_file(self, tmp_path, monkeypatch):
        fake_client = MagicMock()
        fake_client.ping.return_value = True
        fake_client.set.side_effect = RuntimeError("write failed")
        fake_redis = MagicMock()
        fake_redis.from_url.return_value = fake_client
        monkeypatch.setattr("src.caching.REDIS_AVAILABLE", True)
        monkeypatch.setattr("src.caching.redis", fake_redis)
        cache = AggregateCache(use_redis=True, cache_dir=str(tmp_path / "agg"))
        cache.set("fallback-key", {"marker": "test", "n": 1})
        assert cache._file_path("fallback-key").is_file()

    def test_invalidate_redis_and_corrupt_files(self, tmp_path, monkeypatch):
        fake_client = MagicMock()
        fake_client.ping.return_value = True
        fake_client.scan_iter.return_value = ["agent:1"]
        fake_redis = MagicMock()
        fake_redis.from_url.return_value = fake_client
        monkeypatch.setattr("src.caching.REDIS_AVAILABLE", True)
        monkeypatch.setattr("src.caching.redis", fake_redis)
        cache = AggregateCache(use_redis=True, cache_dir=str(tmp_path / "agg"))
        bad = cache.cache_dir / "corrupt.json"
        bad.write_text("x", encoding="utf-8")
        cache.invalidate("agent:")
        fake_client.delete.assert_called()
        assert not bad.exists()
