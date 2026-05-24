"""Tests for summarizer module."""

from __future__ import annotations

from src.summarizer import ActionItem, CallSummarizer, CallSummary


class TestActionItem:
    def test_creation(self):
        a = ActionItem(description="Ring tillbaka kund", responsible="agent", priority="high")
        assert a.responsible == "agent"
        assert a.priority == "high"


class TestCallSummary:
    def test_empty(self):
        cs = CallSummary()
        d = cs.to_dict()
        assert "summary" in d
        assert "action_items" in d
        assert "key_topics" in d
        assert d["backend"] == "extractive"

    def test_with_content(self):
        cs = CallSummary(
            summary="Kort samtal om faktura.",
            summary_sentences=["Kort samtal om faktura."],
            action_items=[ActionItem(description="Skicka ny faktura", responsible="agent")],
            key_topics=["faktura", "betalning"],
            call_outcome="resolved",
            overall_sentiment="positiv",
        )
        d = cs.to_dict()
        assert len(d["action_items"]) == 1
        assert d["call_outcome"] == "resolved"


class TestCallSummarizer:
    def setup_method(self):
        self.cs = CallSummarizer()

    def test_summarize_empty(self):
        result = self.cs.summarize([])
        assert isinstance(result, CallSummary)
        assert result.summary == "Ingen transkribering tillgänglig."

    def test_summarize_basic(self):
        segments = [
            {
                "start": 0,
                "end": 5,
                "text": "Hej, jag har problem med min faktura.",
                "speaker": "SPEAKER_0",
            },
            {
                "start": 5,
                "end": 15,
                "text": "Jag förstår, låt mig kolla upp det. Jag ska återkomma med en lösning.",
                "speaker": "SPEAKER_1",
            },
            {
                "start": 15,
                "end": 25,
                "text": "Tack, då väntar jag på besked. Problemet verkar vara löst nu.",
                "speaker": "SPEAKER_0",
            },
        ]
        result = self.cs.summarize(segments)
        assert len(result.summary_sentences) >= 1
        assert result.call_outcome in ("resolved", "pending", "escalated", "unclear")
        assert len(result.key_topics) >= 0

    def test_extracts_action_items(self):
        segments = [
            {
                "start": 0,
                "end": 10,
                "text": "Jag har problem med min beställning. Agenten sa att hen ska återkomma imorgon och att jag ska skicka in dokumenten.",
            },
        ]
        result = self.cs.summarize(segments)
        assert len(result.action_items) >= 1
        # Should find at least one action item
        descriptions = [a.description for a in result.action_items]
        assert any("återkomma" in d or "skicka" in d for d in descriptions)

    def test_overall_sentiment(self):
        sentiment = [
            {"label": "positiv", "score": 0.9},
            {"label": "positiv", "score": 0.8},
            {"label": "negativ", "score": 0.7},
        ]
        cs = CallSummarizer()
        result = cs._overall_sentiment(sentiment)
        assert result == "positiv"

    def test_determine_outcome_resolved(self):
        cs = CallSummarizer()
        result = cs._determine_outcome("Problemet är löst nu, tack för hjälpen!", None)
        assert result == "resolved"

    def test_determine_outcome_pending(self):
        cs = CallSummarizer()
        result = cs._determine_outcome("Jag inväntar svar från er", None)
        assert result == "pending"
