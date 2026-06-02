"""Tests for the end-to-end call analysis pipeline."""

from __future__ import annotations

from src.pipeline import CallAnalysisPipeline, CallAnalysisReport
from unittest.mock import patch, MagicMock


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

    def test_analyze_segments_with_mistral_flag_accepts_and_merges(self, monkeypatch):
        """Pipeline accepts Mistral flags and surfaces llm in report (even if it falls back)."""
        self._mock_sentiment(monkeypatch)

        # Mock the expensive LLM step so test is fast and deterministic
        fake_llm_out = {
            "trajectory": {"summary": "Test trajectory from Mistral"},
            "actionable_summary": {"problem": "Test problem"},
            "meta": {"model": "mistralai/mistral-medium-3.5", "llm_used": True, "cost_usd": 0.001},
        }

        with patch("src.pipeline.ConversationMistralAnalyzer") as mock_analyzer_cls:
            mock_inst = MagicMock()
            mock_inst.analyze_full_conversation.return_value = fake_llm_out
            mock_analyzer_cls.return_value = mock_inst

            segments = [{"start": 0, "end": 10, "text": "Hej, jag har problem med fakturan.", "speaker": "SPEAKER_1"}]
            # Force the deep path
            report = self.pipe.analyze_segments(
                segments, 
                # We pass via the pipeline instance flags instead of analyze_segments signature for now
            )
            # Re-create with flags by using a fresh pipeline configured for LLM
            pipe_llm = CallAnalysisPipeline(use_mistral_llm=True, llm_model="mistralai/mistral-medium-3.5")
            # Re-apply the sentiment mock on the new instance's path
            monkeypatch.setattr(
                "src.analysis.sentiment.SentimentPipeline.analyze",
                lambda self, texts, **kwargs: [{"label": "negativ", "score": 0.8}] if texts else [],
            )
            report = pipe_llm.analyze_segments(segments)

        assert isinstance(report, CallAnalysisReport)
        assert "llm" in report.to_dict() or hasattr(report, "llm")
        # The llm field should be populated (or contain fallback info)
        assert report.llm is not None
