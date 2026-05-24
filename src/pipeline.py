"""End-to-end call analysis pipeline for Swedish call center conversations.

Orchestrates ASR, sentiment, intent, diarization, summarization, topic modeling,
insights, and predictive analytics into a single unified analysis.

Usage:
    from src.pipeline import CallAnalysisPipeline
    pipe = CallAnalysisPipeline()
    report = pipe.analyze_audio("path/to/call.wav")
    # Or analyze existing segments:
    report = pipe.analyze_segments(segments)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Lazy imports to avoid heavy dependencies at module load time
_SentimentPipeline = None
_IntentClassifier = None
_DiarizationPipeline = None
_CallSummarizer = None
_TopicModeler = None
_InsightsEngine = None
_RiskAnalyzer = None


def _import_sentiment():
    global _SentimentPipeline
    if _SentimentPipeline is None:
        from .sentiment import SentimentPipeline

        _SentimentPipeline = SentimentPipeline
    return _SentimentPipeline


def _import_intent():
    global _IntentClassifier
    if _IntentClassifier is None:
        from .intent import IntentClassifier

        _IntentClassifier = IntentClassifier
    return _IntentClassifier


def _import_diarization():
    global _DiarizationPipeline
    if _DiarizationPipeline is None:
        from .diarization import DiarizationPipeline

        _DiarizationPipeline = DiarizationPipeline
    return _DiarizationPipeline


def _import_summarizer():
    global _CallSummarizer
    if _CallSummarizer is None:
        from .summarizer import CallSummarizer

        _CallSummarizer = CallSummarizer
    return _CallSummarizer


def _import_topic_modeler():
    global _TopicModeler
    if _TopicModeler is None:
        from .topic_modeling import TopicModeler

        _TopicModeler = TopicModeler
    return _TopicModeler


def _import_insights():
    global _InsightsEngine
    if _InsightsEngine is None:
        from .insights import InsightsEngine

        _InsightsEngine = InsightsEngine
    return _InsightsEngine


def _import_predictive():
    global _RiskAnalyzer
    if _RiskAnalyzer is None:
        from .predictive import RiskAnalyzer

        _RiskAnalyzer = RiskAnalyzer
    return _RiskAnalyzer


@dataclass
class CallAnalysisReport:
    """Complete analysis report for a single call."""

    segments: list[dict[str, Any]] = field(default_factory=list)
    sentiment_results: list[dict[str, Any]] = field(default_factory=list)
    intent_results: list[tuple[str, float]] = field(default_factory=list)
    diarization: dict[str, Any] | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    topics: dict[str, Any] = field(default_factory=dict)
    insights: dict[str, Any] = field(default_factory=dict)
    risks: dict[str, Any] = field(default_factory=dict)
    processing_time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "segments": self.segments,
            "sentiment_results": self.sentiment_results,
            "intent_results": [
                {"intent": i, "confidence": round(c, 3)} for i, c in self.intent_results
            ],
            "diarization": self.diarization,
            "summary": self.summary,
            "topics": self.topics,
            "insights": self.insights,
            "risks": self.risks,
            "processing_time_s": self.processing_time_s,
        }


class CallAnalysisPipeline:
    """End-to-end pipeline for analyzing Swedish call center conversations.

    Orchestrates sentiment, intent, diarization, summarization, topics,
    insights, and risk analysis into a single unified report.

    Args:
        sentiment_model: HuggingFace model for sentiment analysis.
        intent_backend: 'heuristic' or 'model' for intent classification.
        diarization_backend: 'heuristic' or 'pyannote' for speaker diarization.
        hf_token: HuggingFace token (required for pyannote diarization).
        device: 'cpu', 'cuda', or 'auto'.
        profile: Sentiment profile (e.g., 'callcenter', 'default').
    """

    def __init__(
        self,
        sentiment_model: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        intent_backend: str = "heuristic",
        diarization_backend: str = "heuristic",
        hf_token: str | None = None,
        device: str = "cpu",
        profile: str = "default",
    ) -> None:
        self.sentiment_model = sentiment_model
        self.intent_backend = intent_backend
        self.diarization_backend = diarization_backend
        self.hf_token = hf_token
        self.device = device
        self.profile = profile

        self._sentiment = None
        self._intent = None
        self._diarization = None
        self._summarizer = None
        self._topic_modeler = None
        self._insights = None
        self._risk = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze_audio(
        self,
        audio_path: str,
        num_speakers: int | None = 2,
        language: str = "sv",
        run_diarization: bool = True,
    ) -> CallAnalysisReport:
        """Analyze a call from an audio file.

        Args:
            audio_path: Path to audio file.
            num_speakers: Expected number of speakers.
            language: Language code for ASR.
            run_diarization: Whether to run speaker diarization.

        Returns:
            CallAnalysisReport with full analysis.
        """
        t0 = time.time()

        # 1. Transcribe audio
        segments = self._transcribe(audio_path, language)

        # 2. Optionally diarize
        diarization_result = None
        if run_diarization:
            diarization_result = self._run_diarization(audio_path, num_speakers)
            segments = self._assign_speakers(segments, diarization_result)

        report = self._analyze_segments(segments)
        report.diarization = diarization_result.to_dict() if diarization_result else None
        report.processing_time_s = round(time.time() - t0, 2)
        return report

    def analyze_segments(self, segments: list[dict[str, Any]]) -> CallAnalysisReport:
        """Analyze pre-existing transcript segments.

        Args:
            segments: List of dicts with 'text' key and optionally 'speaker'.

        Returns:
            CallAnalysisReport with full analysis.
        """
        t0 = time.time()
        report = self._analyze_segments(segments)
        report.processing_time_s = round(time.time() - t0, 2)
        return report

    # ------------------------------------------------------------------
    # Internal pipeline steps
    # ------------------------------------------------------------------
    def _analyze_segments(self, segments: list[dict[str, Any]]) -> CallAnalysisReport:
        """Run full analysis on segments."""
        texts = [s.get("text", "") for s in segments]

        # 1. Sentiment analysis
        sentiment_results = self._run_sentiment(texts)

        # 2. Intent classification
        intent_results = self._run_intent(texts)

        # 3. Summarization
        summary = self._run_summarizer(segments, sentiment_results, intent_results)

        # 4. Topic modeling
        topics = self._run_topic_modeling(segments, sentiment_results)

        # 5. Insights
        insights = self._run_insights(segments, sentiment_results, intent_results, topics)

        # 6. Predictive analytics
        risks = self._run_predictive(sentiment_results, intent_results)

        return CallAnalysisReport(
            segments=segments,
            sentiment_results=sentiment_results,
            intent_results=intent_results,
            summary=summary.to_dict(),
            topics=topics.to_dict(),
            insights=insights.to_dict(),
            risks=risks.to_dict(),
        )

    def _transcribe(self, audio_path: str, language: str) -> list[dict[str, Any]]:
        """Transcribe audio to segments."""
        try:
            from .asr import transcribe

            result = transcribe(
                audio_path,
                model="KBLab/kb-whisper-large",
                language=language,
            )
            return result.get("segments", [])
        except Exception as e:
            logger.error("Transcription failed for %s: %s", audio_path, e)
            return []

    def _run_sentiment(self, texts: list[str]) -> list[dict[str, Any]]:
        """Run sentiment analysis on texts."""
        if self._sentiment is None:
            sentiment_cls = _import_sentiment()
            self._sentiment = sentiment_cls(
                model_name=self.sentiment_model,
                device=self.device,
            )
        try:
            return self._sentiment.analyze(
                texts,
                normalize=True,
                return_all_scores=False,
            )
        except Exception as e:
            logger.error("Sentiment analysis failed: %s", e)
            return [{"label": "neutral", "score": 0.0} for _ in texts]

    def _run_intent(self, texts: list[str]) -> list[tuple[str, float]]:
        """Run intent classification on texts."""
        if self._intent is None:
            intent_cls = _import_intent()
            self._intent = intent_cls(backend=self.intent_backend)
        try:
            return self._intent.classify_batch(texts)
        except Exception as e:
            logger.error("Intent classification failed: %s", e)
            return [("other", 0.0) for _ in texts]

    def _run_diarization(
        self, audio_path: str, num_speakers: int | None
    ) -> Any:
        """Run speaker diarization."""
        if self._diarization is None:
            diarization_cls = _import_diarization()
            self._diarization = diarization_cls(
                backend=self.diarization_backend,
                hf_token=self.hf_token,
                device=self.device,
            )
        try:
            return self._diarization.diarize(audio_path, num_speakers=num_speakers)
        except Exception as e:
            logger.error("Diarization failed for %s: %s", audio_path, e)
            return None

    def _assign_speakers(
        self,
        segments: list[dict[str, Any]],
        diarization_result: Any,
    ) -> list[dict[str, Any]]:
        """Assign speaker labels to ASR segments."""
        if diarization_result is None or not diarization_result.segments:
            return segments
        return self._diarization.assign_speakers_to_segments(
            segments, diarization_result
        )

    def _run_summarizer(
        self,
        segments: list[dict[str, Any]],
        sentiment_results: list[dict[str, Any]],
        intent_results: list[tuple[str, float]],
    ) -> Any:
        """Generate call summary."""
        if self._summarizer is None:
            summarizer_cls = _import_summarizer()
            self._summarizer = summarizer_cls()
        try:
            return self._summarizer.summarize(
                segments,
                sentiment_results=sentiment_results,
                intent_results=intent_results,
            )
        except Exception as e:
            logger.error("Summarization failed: %s", e)
            from .summarizer import CallSummary

            return CallSummary()

    def _run_topic_modeling(
        self,
        segments: list[dict[str, Any]],
        sentiment_results: list[dict[str, Any]],
    ) -> Any:
        """Extract topics from segments."""
        if self._topic_modeler is None:
            topic_modeler_cls = _import_topic_modeler()
            self._topic_modeler = topic_modeler_cls()
        try:
            return self._topic_modeler.extract_topics(
                segments, sentiment_results=sentiment_results
            )
        except Exception as e:
            logger.error("Topic modeling failed: %s", e)
            from .topic_modeling import TopicReport

            return TopicReport()

    def _run_insights(
        self,
        segments: list[dict[str, Any]],
        sentiment_results: list[dict[str, Any]],
        intent_results: list[tuple[str, float]],
        topics: Any,
    ) -> Any:
        """Generate insights."""
        if self._insights is None:
            insights_cls = _import_insights()
            self._insights = insights_cls()
        try:
            return self._insights.analyze(
                segments,
                intent_results=intent_results,
                sentiment_results=sentiment_results,
                topics=topics.topics if hasattr(topics, "topics") else None,
            )
        except Exception as e:
            logger.error("Insights generation failed: %s", e)
            from .insights import InsightsReport

            return InsightsReport()

    def _run_predictive(
        self,
        sentiment_results: list[dict[str, Any]],
        intent_results: list[tuple[str, float]],
    ) -> Any:
        """Run predictive risk analysis."""
        if self._risk is None:
            risk_cls = _import_predictive()
            self._risk = risk_cls()
        try:
            return self._risk.analyze(
                sentiment_results=sentiment_results,
                intent_results=intent_results,
            )
        except Exception as e:
            logger.error("Predictive analysis failed: %s", e)
            from .predictive import RiskAssessment

            return RiskAssessment()


__all__ = ["CallAnalysisPipeline", "CallAnalysisReport"]
