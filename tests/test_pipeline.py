"""Tests for the end-to-end call analysis pipeline."""

from __future__ import annotations

from src.pipeline import CallAnalysisPipeline, CallAnalysisReport


class TestCallAnalysisPipeline:
    def setup_method(self):
        self.pipe = CallAnalysisPipeline()

    def _mock_sentiment(self, monkeypatch):
        """Mock sentiment pipeline to avoid loading real models."""

        def mock_analyze(self, texts, **kwargs):
            return (
                [
                    {"label": "negativ", "score": 0.8},
                    {"label": "neutral", "score": 0.6},
                    {"label": "positiv", "score": 0.9},
                ][: max(1, len(texts))]
                if texts
                else []
            )

        monkeypatch.setattr(
            "src.analysis.sentiment.SentimentPipeline.analyze",
            mock_analyze,
        )

    def test_analyze_segments_basic(self, monkeypatch):
        self._mock_sentiment(monkeypatch)
        segments = [
            {
                "start": 0,
                "end": 5,
                "text": "Jag är mycket missnöjd med fakturan.",
                "speaker": "SPEAKER_0",
            },
            {"start": 5, "end": 10, "text": "Jag ska hjälpa dig direkt.", "speaker": "SPEAKER_1"},
            {"start": 10, "end": 15, "text": "Tack, det uppskattar jag.", "speaker": "SPEAKER_0"},
        ]
        report = self.pipe.analyze_segments(segments)

        assert isinstance(report, CallAnalysisReport)
        assert len(report.segments) == 3
        assert len(report.sentiment_results) == 3
        assert len(report.intent_results) == 3
        assert report.processing_time_s >= 0

        # Verify structure – each field is a dict from the respective module
        assert isinstance(report.summary, dict)
        assert isinstance(report.topics, dict)
        assert isinstance(report.insights, dict)
        assert isinstance(report.risks, dict)

    def test_analyze_segments_empty(self, monkeypatch):
        self._mock_sentiment(monkeypatch)
        report = self.pipe.analyze_segments([])
        assert isinstance(report, CallAnalysisReport)
        assert report.segments == []
        assert report.sentiment_results == []
        assert report.intent_results == []

    def test_report_to_dict(self, monkeypatch):
        self._mock_sentiment(monkeypatch)
        segments = [
            {"start": 0, "end": 5, "text": "Bra service!"},
        ]
        report = self.pipe.analyze_segments(segments)
        d = report.to_dict()

        assert "segments" in d
        assert "sentiment_results" in d
        assert "intent_results" in d
        assert "summary" in d
        assert "topics" in d
        assert "insights" in d
        assert "risks" in d
        assert "processing_time_s" in d

        # Intent results should be serialized as dicts
        assert isinstance(d["intent_results"], list)
        if d["intent_results"]:
            assert "intent" in d["intent_results"][0]
            assert "confidence" in d["intent_results"][0]

    def test_analyze_audio_missing_file(self, monkeypatch):
        """Should gracefully handle missing audio files."""
        self._mock_sentiment(monkeypatch)
        report = self.pipe.analyze_audio("/nonexistent/audio.wav")
        assert isinstance(report, CallAnalysisReport)
        assert report.segments == []
        # Diarization is attempted even when transcription fails; it returns an empty result
        assert report.diarization is not None
        assert report.diarization.get("segments") == []
