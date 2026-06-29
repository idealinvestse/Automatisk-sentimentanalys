"""Fas 1 edge case tests: long calls, low confidence, endpoint errors."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import app as api_app
from src.api.settings import get_api_settings
from src.pipeline import CallAnalysisPipeline

client = TestClient(api_app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _clear_api_settings_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SENTIMENT_API_KEY", raising=False)
    get_api_settings.cache_clear()


def _mock_low_confidence_sentiment(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_analyze(self, texts, **kwargs):
        return [{"label": "neutral", "score": 0.45} for _ in texts]

    monkeypatch.setattr(
        "src.analysis.sentiment.SentimentPipeline.analyze",
        mock_analyze,
    )


class TestLongCallEdgeCase:
    def test_analyze_segments_many_segments_no_crash(self, monkeypatch):
        _mock_low_confidence_sentiment(monkeypatch)
        segments = [
            {
                "start": float(i * 2),
                "end": float(i * 2 + 2),
                "text": f"Segment {i} om faktura och support.",
                "speaker": "SPEAKER_0" if i % 2 == 0 else "SPEAKER_1",
            }
            for i in range(55)
        ]
        pipe = CallAnalysisPipeline(profile="callcenter")
        report = pipe.analyze_segments(segments)
        assert len(report.segments) == 55
        assert "agent_performance" in (report.results or {})


class TestLowConfidenceEdgeCase:
    def test_pipeline_handles_low_confidence_without_crash(self, monkeypatch):
        _mock_low_confidence_sentiment(monkeypatch)
        pipe = CallAnalysisPipeline(profile="callcenter", use_mistral_llm=False)
        segments = [
            {"start": 0, "end": 2, "text": "Kanske okej kanske inte.", "speaker": "C"},
            {"start": 2, "end": 5, "text": "Vi får se.", "speaker": "A"},
        ]
        report = pipe.analyze_segments(segments)
        assert report.sentiment_results
        assert isinstance(report.results, dict)


class TestFas4EndpointErrors:
    def test_qa_score_empty_segments_422(self):
        r = client.post("/qa/score", json={"segments": []})
        assert r.status_code == 422

    def test_semantic_search_missing_query_422(self):
        r = client.post("/search/semantic", json={"segments_list": [[{"text": "x"}]]})
        assert r.status_code == 422

    def test_alerts_empty_body_422(self):
        r = client.post("/alerts", json={})
        assert r.status_code == 422

    def test_agent_performance_invalid_id_422(self):
        r = client.post(
            "/agent_performance/bad id!",
            json={"segments_list": [[{"text": "Hej"}]], "agent_id": "bad id!"},
        )
        assert r.status_code == 422

    def test_hot_topics_too_many_calls_422(self):
        too_many = [[{"text": "x"}] for _ in range(51)]
        r = client.post("/insights/hot_topics", json={"segments_list": too_many})
        assert r.status_code == 422


class TestCacheInvalidationEdgeCase:
    def test_invalidate_clears_cached_aggregate(self, monkeypatch, tmp_path):
        _mock_low_confidence_sentiment(monkeypatch)
        from src.caching import AggregateCache

        pipe = CallAnalysisPipeline(profile="callcenter")
        pipe.cache = AggregateCache(cache_dir=str(tmp_path / "agg"))
        segs = [{"start": 0, "end": 2, "text": "Faktura A", "speaker": "C"}]
        report = pipe.analyze_segments(segs)
        m1 = pipe.get_cached_agent_performance("Agent-X", [report])
        m2 = pipe.get_cached_agent_performance("Agent-X", [report])
        assert m1["cache_hit"] is False
        assert m2["cache_hit"] is True
        pipe.invalidate_aggregate_cache("Agent-X")
        m3 = pipe.get_cached_agent_performance("Agent-X", [report])
        assert m3["cache_hit"] is False


class TestPIIWithLLMPath:
    def test_pii_redaction_before_llm_no_crash(self, monkeypatch):
        _mock_low_confidence_sentiment(monkeypatch)
        monkeypatch.setattr(
            "src.profiles.resolve_profile",
            lambda *a, **k: ("callcenter", {"llm": {"anonymize_before_llm": True}}),
            raising=False,
        )
        monkeypatch.setattr(
            "src.llm.mistral_analyzer.ConversationMistralAnalyzer.analyze_full_conversation",
            lambda self, **kwargs: {"fallback": True, "meta": {"llm_used": False}},
        )
        pipe = CallAnalysisPipeline(profile="callcenter", use_mistral_llm=True)
        segments = [
            {
                "start": 0,
                "end": 3,
                "text": "Mitt personnummer är 19850101-1234 och mail test@example.com",
                "speaker": "C",
            },
            {"start": 3, "end": 6, "text": "Tack, jag hjälper dig.", "speaker": "A"},
        ]
        report = pipe.analyze_segments(segments)
        assert "pii_redaction" in (report.results or {})
        pl = report.results["pii_redaction"]
        assert pl.get("total_redacted", 0) >= 1
