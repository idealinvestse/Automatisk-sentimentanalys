"""End-to-end call analysis pipeline for Swedish call center conversations.

Orchestrates ASR and text analysis through modular components.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .analysis import run_analyzers
from .core.models import AnalysisContext, CallAnalysisReport, Segment
from .transcription import get_transcriber

logger = logging.getLogger(__name__)


class CallAnalysisPipeline:
    """End-to-end pipeline for analyzing Swedish call center conversations.

    Orchestrates transcription and modular text analysis.

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
        diarization_backend: str = "heuristic",  # Kept for backward compatibility
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

    def analyze_audio(
        self,
        audio_path: str,
        num_speakers: int | None = 2,
        language: str = "sv",
        run_diarization: bool = True,
        selected_analyzers: list[str] | None = None,
    ) -> CallAnalysisReport:
        """Analyze a call from an audio file.

        Args:
            audio_path: Path to audio file.
            num_speakers: Expected number of speakers.
            language: Language code for ASR.
            run_diarization: Whether to run speaker diarization.
            selected_analyzers: Optional list of analyzer names to run. Runs all by default.

        Returns:
            CallAnalysisReport with full analysis.
        """
        t0 = time.time()

        # 1. Transcribe and optionally diarize
        try:
            # Obtain transcriber (defaults to faster-whisper)
            transcriber = get_transcriber(
                backend="faster",
                model_name="kb-whisper-large",
                device=self.device,
            )
            transcript = transcriber.transcribe(
                audio_path=audio_path,
                language=language,
                diarize=run_diarization,
                num_speakers=num_speakers,
            )
        except Exception as e:
            logger.error("Transcription or ASR initialization failed for %s: %s", audio_path, e)
            from .core.models import Transcript

            transcript = Transcript(
                model="kb-whisper-large",
                backend="faster",
                language=language,
                duration=0.0,
                processing_time=0.0,
                segments=[],
                diarization={"segments": [], "backend": "failed", "error": str(e)},
            )

        # 2. Create context and run text analysis
        ctx = AnalysisContext(
            transcript=transcript,
            segments=transcript.segments,
        )

        results = run_analyzers(ctx, selected=selected_analyzers)
        proc_time = round(time.time() - t0, 2)

        # 4. Construct backwards-compatible report
        segments_dict = [s.to_dict() for s in transcript.segments]
        diarization_data = transcript.diarization

        return CallAnalysisReport(
            segments=segments_dict,
            sentiment_results=results.get("sentiment", []),
            intent_results=results.get("intent", []),
            diarization=diarization_data,
            summary=results.get("summary", {}),
            topics=results.get("topics", {}),
            insights=results.get("insights", {}),
            risks=results.get(
                "predictive", {}
            ),  # Map predictive risks to risks for backward-compatibility
            processing_time_s=proc_time,
            results=results,
        )

    def analyze_segments(
        self,
        segments: list[dict[str, Any]],
        selected_analyzers: list[str] | None = None,
    ) -> CallAnalysisReport:
        """Analyze pre-existing transcript segments.

        Args:
            segments: List of dicts with 'text' key and optionally 'speaker'.
            selected_analyzers: Optional list of analyzer names to run. Runs all by default.

        Returns:
            CallAnalysisReport with full analysis.
        """
        t0 = time.time()

        # Convert segment dicts to Segment dataclasses
        typed_segments = []
        for s in segments:
            typed_segments.append(
                Segment(
                    start=float(s.get("start", 0.0) or 0.0),
                    end=float(s.get("end", 0.0) or 0.0),
                    text=str(s.get("text", "")),
                    speaker=s.get("speaker"),
                    avg_confidence=s.get("avg_confidence"),
                )
            )

        # Create context and run text analysis
        ctx = AnalysisContext(
            transcript=None,
            segments=typed_segments,
        )

        results = run_analyzers(ctx, selected=selected_analyzers)
        proc_time = round(time.time() - t0, 2)

        return CallAnalysisReport(
            segments=segments,
            sentiment_results=results.get("sentiment", []),
            intent_results=results.get("intent", []),
            diarization=None,
            summary=results.get("summary", {}),
            topics=results.get("topics", {}),
            insights=results.get("insights", {}),
            risks=results.get(
                "predictive", {}
            ),  # Map predictive risks to risks for backward-compatibility
            processing_time_s=proc_time,
            results=results,
        )
