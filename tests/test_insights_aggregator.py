"""Tests for InsightsAggregator trend logic (B4)."""

from __future__ import annotations

from src.insights_aggregator import InsightsAggregator, _sentiment_trend_label


class TestSentimentTrendLabel:
    def test_empty_is_stable(self) -> None:
        assert _sentiment_trend_label([]) == "stable"

    def test_short_series_uses_average(self) -> None:
        assert _sentiment_trend_label([0.5, 0.6]) == "up"
        assert _sentiment_trend_label([-0.5, -0.6]) == "down"

    def test_longer_series_compares_halves(self) -> None:
        improving = [-0.5, -0.4, -0.3, 0.3, 0.4, 0.5]
        declining = [0.5, 0.4, 0.3, -0.3, -0.4, -0.5]
        assert _sentiment_trend_label(improving) == "up"
        assert _sentiment_trend_label(declining) == "down"
        assert _sentiment_trend_label([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) == "stable"


class TestInsightsAggregatorAggregate:
    def test_hot_topic_trend_from_ordered_sentiments(self) -> None:
        reports = [
            {
                "sentiment_results": [{"label": "negativ"}],
                "topics": {"topics": [{"topic": "fakturering"}]},
                "segments": [{"text": "Min faktura är fel", "speaker": "SPEAKER_0"}],
            },
            {
                "sentiment_results": [{"label": "negativ"}],
                "topics": {"topics": [{"topic": "fakturering"}]},
                "segments": [{"text": "Fortfarande fel faktura", "speaker": "SPEAKER_0"}],
            },
            {
                "sentiment_results": [{"label": "negativ"}],
                "topics": {"topics": [{"topic": "fakturering"}]},
                "segments": [{"text": "Fortfarande väntar", "speaker": "SPEAKER_0"}],
            },
            {
                "sentiment_results": [{"label": "positiv"}],
                "topics": {"topics": [{"topic": "fakturering"}]},
                "segments": [{"text": "Nu är fakturan rätt, tack!", "speaker": "SPEAKER_0"}],
            },
            {
                "sentiment_results": [{"label": "positiv"}],
                "topics": {"topics": [{"topic": "fakturering"}]},
                "segments": [{"text": "Mycket nöjd nu", "speaker": "SPEAKER_0"}],
            },
        ]
        agg = InsightsAggregator().aggregate(reports, min_volume=2)
        hot = {h.topic: h for h in agg.hot_topics}
        assert "fakturering" in hot
        assert hot["fakturering"].trend == "up"
