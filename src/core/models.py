"""Shared data models for the Automatic Sentiment Analysis system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Word:
    """Represents a single transcribed word with timestamp and confidence."""

    start: float
    end: float
    word: str
    prob: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "word": self.word,
            "prob": self.prob,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Word:
        return cls(
            start=float(data.get("start", 0.0)),
            end=float(data.get("end", 0.0)),
            word=str(data.get("word", "")),
            prob=float(data.get("prob", 0.0)),
        )


@dataclass
class Segment:
    """Represents a segment of a conversation, with optional word-level details."""

    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    speaker: str | None = None
    avg_confidence: float | None = None
    # Task 1.2: explicit confidence + low_confidence flag for downstream
    # (e.g. higher lexicon weight on uncertain ASR segments). Stored both
    # at top level (for easy access) and in properties for forward compat.
    confidence: float | None = None
    low_confidence: bool = False
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        res: dict[str, Any] = {
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }
        if self.words:
            res["words"] = [w.to_dict() for w in self.words]
        if self.speaker is not None:
            res["speaker"] = self.speaker
        if self.avg_confidence is not None:
            res["avg_confidence"] = self.avg_confidence
        if self.confidence is not None:
            res["confidence"] = self.confidence
        if self.low_confidence:
            res["low_confidence"] = True
        if self.properties:
            res["properties"] = self.properties
        return res

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Segment:
        words_data = data.get("words", [])
        words = [Word.from_dict(w) for w in words_data] if words_data else []
        return cls(
            start=float(data.get("start", 0.0)),
            end=float(data.get("end", 0.0)),
            text=str(data.get("text", "")),
            words=words,
            speaker=data.get("speaker"),
            avg_confidence=data.get("avg_confidence"),
            confidence=data.get("confidence") or data.get("avg_confidence"),
            low_confidence=bool(data.get("low_confidence", False)),
            properties=data.get("properties", {}),
        )


@dataclass
class Transcript:
    """Represents the complete transcription output of an audio file."""

    model: str
    backend: str
    language: str
    duration: float | None
    processing_time: float
    segments: list[Segment] = field(default_factory=list)
    revision: str | None = None
    diarization: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        res: dict[str, Any] = {
            "model": self.model,
            "backend": self.backend,
            "language": self.language,
            "duration": self.duration,
            "processing_time": self.processing_time,
            "segments": [s.to_dict() for s in self.segments],
        }
        if self.revision:
            res["revision"] = self.revision
        if self.diarization:
            res["diarization"] = self.diarization
        return res

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Transcript:
        segs_data = data.get("segments", [])
        segments = [Segment.from_dict(s) for s in segs_data]
        return cls(
            model=str(data.get("model", "")),
            backend=str(data.get("backend", "")),
            language=str(data.get("language", "sv")),
            duration=data.get("duration"),
            processing_time=float(data.get("processing_time", 0.0)),
            segments=segments,
            revision=data.get("revision"),
            diarization=data.get("diarization"),
        )


@dataclass
class AnalysisContext:
    """Context passed to all analyzers during pipeline execution."""

    transcript: Transcript | None = None
    segments: list[Segment] = field(default_factory=list)
    results: dict[str, Any] = field(default_factory=dict)


@dataclass
class CallAnalysisReport:
    """Unified and backwards-compatible analysis report for a call."""

    segments: list[dict[str, Any]] = field(default_factory=list)
    sentiment_results: list[dict[str, Any]] = field(default_factory=list)
    intent_results: list[tuple[str, float]] = field(default_factory=list)
    diarization: dict[str, Any] | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    topics: dict[str, Any] = field(default_factory=dict)
    insights: dict[str, Any] = field(default_factory=dict)
    risks: dict[str, Any] = field(default_factory=dict)
    processing_time_s: float = 0.0
    results: dict[str, Any] = field(
        default_factory=dict
    )  # Stores arbitrary model outputs dynamically

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary, maintaining backward compatibility."""
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
            "results": self.results,
        }
