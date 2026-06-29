"""Tests for Edge AI offline inference."""

from __future__ import annotations

from src.edge.local_inference import analyze_segments_offline, analyze_text_offline


def test_analyze_text_offline(monkeypatch):
    def fake_smart(texts, **kwargs):
        return ([{"label": "positiv", "score": 0.9}], {"profile": "callcenter"})

    monkeypatch.setattr("src.sentiment.analyze_smart", fake_smart)
    monkeypatch.setattr(
        "src.intent.IntentClassifier.classify",
        lambda self, text: ("support", 0.8),
    )
    result = analyze_text_offline("Tack för hjälpen!", profile="callcenter")
    assert result.offline is True
    assert result.llm_used is False
    assert len(result.segments) == 1
    assert result.segments[0].sentiment_label == "positiv"


def test_analyze_segments_offline(monkeypatch):
    monkeypatch.setattr(
        "src.sentiment.analyze_smart",
        lambda texts, **kwargs: ([{"label": "neutral", "score": 0.5}] * len(texts), {}),
    )
    monkeypatch.setattr(
        "src.intent.IntentClassifier.classify",
        lambda self, text: ("info", 0.7),
    )
    segments = [{"text": "Hej", "start": 0, "end": 1}]
    result = analyze_segments_offline(segments, profile="callcenter")
    assert len(result.segments) == 1
    assert "No LLM" in result.limitations[0]
