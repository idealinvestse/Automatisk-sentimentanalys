"""Tests for topic_modeling, insights, and predictive modules."""

from __future__ import annotations

from src.insights import InsightsEngine, InsightsReport
from src.predictive import RiskAnalyzer, RiskAssessment
from src.topic_modeling import TopicModeler, TopicReport, TopicResult


class TestTopicModeler:
    def setup_method(self):
        self.tm = TopicModeler()

    def test_extract_topics(self):
        segments = [
            {"start": 0, "end": 5, "text": "Jag har problem med min faktura, den är för hög."},
            {"start": 5, "end": 10, "text": "Kan ni hjälpa mig med betalningen?"},
            {"start": 10, "end": 15, "text": "Min leverans har inte kommit än."},
        ]
        report = self.tm.extract_topics(segments)
        assert isinstance(report, TopicReport)
        assert len(report.topics) >= 1
        assert report.total_segments == 3

    def test_extract_topics_with_sentiment(self):
        segments = [
            {"start": 0, "end": 5, "text": "Fakturan är fel."},
        ]
        sentiment = [{"label": "negativ", "score": 0.8}]
        report = self.tm.extract_topics(segments, sentiment_results=sentiment)
        assert len(report.topics) >= 1
        if report.topics:
            assert "negativ" in report.topics[0].sentiment_distribution

    def test_empty_segments(self):
        report = self.tm.extract_topics([])
        assert len(report.topics) == 0

    def test_emerging_topics(self):
        segments = [
            {"start": 0, "end": 5, "text": "Nya portalen krånglar hela tiden."},
            {"start": 5, "end": 10, "text": "Portalen är seg och portalen fungerar inte."},
            {"start": 10, "end": 15, "text": "Portalen igen – samma problem som igår."},
        ]
        report = self.tm.extract_topics(segments)
        assert "portalen" in report.emerging_topics

    def test_topic_trends(self):
        r1 = TopicReport(
            topics=[TopicResult(name="faktura", frequency=3)],
            timestamp="2026-05-01T00:00:00Z",
        )
        r2 = TopicReport(
            topics=[TopicResult(name="faktura", frequency=5)],
            timestamp="2026-05-02T00:00:00Z",
        )
        trends = self.tm.topic_trends([r1, r2])
        assert len(trends) == 1
        assert trends[0]["topic"] == "faktura"


class TestInsightsEngine:
    def setup_method(self):
        self.ie = InsightsEngine()

    def test_analyze_basic(self):
        segments = [
            {"start": 0, "end": 5, "text": "Jag är missnöjd.", "speaker": "SPEAKER_0"},
            {
                "start": 5,
                "end": 10,
                "text": "Jag förstår, jag ska hjälpa dig.",
                "speaker": "SPEAKER_1",
            },
        ]
        intent = [("complaint", 0.9), ("information_request", 0.7)]
        sentiment = [{"label": "negativ", "score": 0.9}, {"label": "neutral", "score": 0.6}]
        report = self.ie.analyze(segments, intent_results=intent, sentiment_results=sentiment)
        assert isinstance(report, InsightsReport)
        assert len(report.key_findings) >= 1

    def test_detects_risks(self):
        sentiment = [{"label": "negativ", "score": 0.9}] * 10
        intent = [("complaint", 0.9)] * 5
        report = self.ie.analyze([], intent_results=intent, sentiment_results=sentiment)
        assert len(report.risk_alerts) >= 1
        assert any("CHURN" in a or "ESCALATION" in a for a in report.risk_alerts)

    def test_empty_input(self):
        report = self.ie.analyze([])
        assert isinstance(report, InsightsReport)
        assert report.root_causes == []
        assert report.key_findings == []


class TestRiskAnalyzer:
    def setup_method(self):
        self.ra = RiskAnalyzer()

    def test_low_risk(self):
        sentiment = [{"label": "positiv", "score": 0.9}] * 5
        intent = [("information_request", 0.8)] * 5
        result = self.ra.analyze(sentiment_results=sentiment, intent_results=intent)
        assert result.risk_level == "low"
        assert result.churn_risk < 0.3

    def test_high_churn_risk(self):
        sentiment = [{"label": "negativ", "score": 0.9}] * 10
        intent = [("cancellation", 0.9)] * 3
        result = self.ra.analyze(sentiment_results=sentiment, intent_results=intent)
        assert result.churn_risk > 0.5
        assert result.risk_level in ("high", "critical")

    def test_satisfaction_score(self):
        sentiment = [{"label": "positiv", "score": 0.9}] * 8 + [
            {"label": "negativ", "score": 0.9}
        ] * 2
        result = self.ra.analyze(sentiment_results=sentiment)
        assert result.satisfaction_score > 0.6

    def test_risk_assessment_to_dict(self):
        ra = RiskAssessment(churn_risk=0.7, escalation_risk=0.3, risk_level="high")
        d = ra.to_dict()
        assert d["churn_risk"] == 0.7
        assert d["risk_level"] == "high"
