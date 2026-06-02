"""Pydantic request and response models for the Swedish Sentiment API.

All schemas include field-level validation where it adds value (non-empty lists,
file/directory existence, device string format, etc.).
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# /analyze
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    texts: list[str] = Field(..., description="List of texts to analyze")
    datatype: str | None = Field(None, description="Data type: post, comment, article, review, ...")
    source: str | None = Field(None, description="Source: forum, magazine, news, social, ...")
    profile: str | None = Field(None, description="Explicit profile name to use")
    model: str | None = Field(None, description="Optional model override")
    device: str | None = Field("auto", description="Device: auto, cpu, cuda, cuda:0, mps")
    batch_size: int = Field(16, ge=1, le=128)
    return_all_scores: bool = Field(False)
    max_length: int | None = Field(None, ge=8, le=4096)
    clean: bool = Field(True)
    normalize: bool = Field(True)
    lexicon_file: str | None = Field(
        None,
        description="Path to Swedish lexicon (CSV/TSV) with columns term|word and polarity|score|sentiment",
    )
    lexicon_weight: float = Field(0.0, ge=0.0, le=1.0, description="Blend weight [0..1]")

    @field_validator("texts")
    @classmethod
    def texts_must_not_be_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("texts must not be empty")
        return v


class AnalyzeResponse(BaseModel):
    meta: dict[str, Any]
    timestamp: str
    results: list[Any]


# ---------------------------------------------------------------------------
# /transcribe
# ---------------------------------------------------------------------------


class TranscribeRequest(BaseModel):
    audio_path: str = Field(..., description="Path to audio file accessible by the server")
    model: str = Field("kb-whisper-large")
    backend: str = Field("faster", description="faster | transformers")
    device: str = Field("auto")
    language: str = Field("sv")
    beam_size: int = Field(5, ge=1, le=10)
    vad: bool = Field(True)
    word_timestamps: bool = Field(True)
    chunk_length_s: int = Field(30, ge=5, le=60)
    revision: str | None = Field(None, description="KB-Whisper revision: standard|strict|subtitle")
    diarize: bool = Field(False, description="Run speaker diarization")
    num_speakers: int | None = Field(None, description="Expected number of speakers (None=auto)")

    @field_validator("audio_path")
    @classmethod
    def audio_path_must_exist(cls, v: str) -> str:
        if not os.path.isfile(v):
            raise ValueError(f"Audio file not found on server: {v!r}")
        return v


class TranscribeResponse(BaseModel):
    transcript: dict[str, Any]
    timestamp: str


# ---------------------------------------------------------------------------
# /analyze_conversation
# ---------------------------------------------------------------------------


class AnalyzeConversationRequest(BaseModel):
    audio_path: str = Field(..., description="Path to audio file accessible by the server")
    model: str = Field("kb-whisper-large")
    backend: str = Field("faster")
    device: str = Field("auto")
    language: str = Field("sv")
    beam_size: int = Field(5, ge=1, le=10)
    vad: bool = Field(True)
    word_timestamps: bool = Field(False)
    chunk_length_s: int = Field(30, ge=5, le=60)
    revision: str | None = Field(None, description="KB-Whisper revision: standard|strict|subtitle")
    diarize: bool = Field(False, description="Run speaker diarization")
    num_speakers: int | None = Field(None, description="Expected number of speakers")
    return_all_scores: bool = Field(True)
    sentiment_model: str | None = Field(None, description="Optional override for sentiment model")
    lexicon_file: str | None = Field(None)
    lexicon_weight: float = Field(0.0, ge=0.0, le=1.0)

    @field_validator("audio_path")
    @classmethod
    def audio_path_must_exist(cls, v: str) -> str:
        if not os.path.isfile(v):
            raise ValueError(f"Audio file not found on server: {v!r}")
        return v


class SegmentSentiment(BaseModel):
    index: int
    start: float | None
    end: float | None
    text: str
    label: str
    score: float
    negativ: float | None = None
    neutral: float | None = None
    positiv: float | None = None
    intent: str | None = None
    intent_confidence: float | None = None


class AnalyzeConversationResponse(BaseModel):
    transcript: dict[str, Any]
    segment_sentiments: list[SegmentSentiment]
    meta: dict[str, Any]
    timestamp: str


# ---------------------------------------------------------------------------
# /analyze_pipeline
# ---------------------------------------------------------------------------


class PipelineRequest(BaseModel):
    """Request for the full call analysis pipeline."""

    segments: list[dict[str, Any]] = Field(
        ...,
        description="ASR segments with 'text' and optionally 'speaker' keys",
    )
    sentiment_model: str | None = Field(None, description="Optional sentiment model override")
    device: str = Field("auto")

    @field_validator("segments")
    @classmethod
    def segments_must_not_be_empty(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not v:
            raise ValueError("segments must not be empty")
        return v


class PipelineResponse(BaseModel):
    """Response from the full call analysis pipeline."""

    sentiment_results: list[dict[str, Any]]
    intent_results: list[dict[str, Any]]
    summary: dict[str, Any]
    topics: dict[str, Any]
    insights: dict[str, Any]
    risks: dict[str, Any]
    processing_time_s: float
    timestamp: str


# ---------------------------------------------------------------------------
# /batch_transcribe
# ---------------------------------------------------------------------------


class BatchTranscribeRequest(BaseModel):
    audio_paths: list[str] | None = None
    directory: str | None = None
    glob: str | None = Field(None, description="Glob pattern within directory, e.g. **/*.wav")
    recursive: bool = True
    limit: int | None = Field(None, ge=1)
    workers: int = Field(1, ge=1, le=8)
    worker_timeout: float = Field(300.0, gt=0.0, description="Per-file worker timeout in seconds")
    # ASR params
    model: str = Field("kb-whisper-large")
    backend: str = Field("faster")
    device: str = Field("auto")
    language: str = Field("sv")
    beam_size: int = Field(5, ge=1, le=10)
    vad: bool = Field(True)
    word_timestamps: bool = Field(True)
    chunk_length_s: int = Field(30, ge=5, le=60)
    revision: str | None = Field(None, description="KB-Whisper revision: standard|strict|subtitle")
    diarize: bool = Field(False, description="Run speaker diarization")
    num_speakers: int | None = Field(None, description="Expected number of speakers")


class BatchTranscribeItem(BaseModel):
    file: str
    transcript: dict[str, Any] | None = None
    error: str | None = None


class BatchTranscribeResponse(BaseModel):
    items: list[BatchTranscribeItem]
    ok: int
    failed: int
    total: int
    timestamp: str


# ---------------------------------------------------------------------------
# /batch_analyze_conversation
# ---------------------------------------------------------------------------


class BatchAnalyzeConversationRequest(BaseModel):
    audio_paths: list[str] | None = None
    directory: str | None = None
    glob: str | None = Field(None)
    recursive: bool = True
    limit: int | None = Field(None, ge=1)
    workers: int = Field(1, ge=1, le=8)
    worker_timeout: float = Field(300.0, gt=0.0, description="Per-file worker timeout in seconds")
    # ASR
    model: str = Field("kb-whisper-large")
    backend: str = Field("faster")
    device: str = Field("auto")
    language: str = Field("sv")
    beam_size: int = Field(5, ge=1, le=10)
    vad: bool = Field(True)
    word_timestamps: bool = Field(False)
    chunk_length_s: int = Field(30, ge=5, le=60)
    revision: str | None = Field(None, description="KB-Whisper revision: standard|strict|subtitle")
    diarize: bool = Field(False, description="Run speaker diarization")
    num_speakers: int | None = Field(None, description="Expected number of speakers")
    # Sentiment
    sentiment_model: str | None = Field(None)
    sentiment_batch_size: int = Field(16, ge=1, le=128, description="Batch size for sentiment inference")
    lexicon_file: str | None = Field(None)
    lexicon_weight: float = Field(0.0, ge=0.0, le=1.0)


class BatchAnalyzeConversationItem(BaseModel):
    file: str
    transcript: dict[str, Any] | None = None
    segment_sentiments: list[SegmentSentiment] | None = None
    meta: dict[str, Any] | None = None
    error: str | None = None


class BatchAnalyzeConversationResponse(BaseModel):
    items: list[BatchAnalyzeConversationItem]
    ok: int
    failed: int
    total: int
    timestamp: str


# ---------------------------------------------------------------------------
# /scan_process
# ---------------------------------------------------------------------------


class ScanProcessRequest(BaseModel):
    directory: str = Field(..., description="Directory to scan")
    pattern: str | None = Field(
        None, description="Glob pattern relative to directory (e.g., **/*.wav)"
    )
    recursive: bool = True
    batch_size: int = Field(4, ge=1, le=64, description="Number of files per processing batch")
    max_files: int | None = Field(None, ge=1)
    state_file: str | None = Field(None, description="Optional JSON file to track processed files")
    workers: int = Field(1, ge=1, le=8, description="Parallel workers per batch")
    worker_timeout: float = Field(300.0, gt=0.0, description="Per-file worker timeout in seconds")
    operation: str = Field("transcribe", description="transcribe | analyze_conversation")
    # ASR
    model: str = Field("kb-whisper-large")
    backend: str = Field("faster")
    device: str = Field("auto")
    language: str = Field("sv")
    beam_size: int = Field(5, ge=1, le=10)
    vad: bool = Field(True)
    word_timestamps: bool = Field(False)
    chunk_length_s: int = Field(30, ge=5, le=60)
    revision: str | None = Field(None, description="KB-Whisper revision: standard|strict|subtitle")
    diarize: bool = Field(False, description="Run speaker diarization")
    num_speakers: int | None = Field(None, description="Expected number of speakers")
    # Sentiment (used when operation=analyze_conversation)
    sentiment_model: str | None = Field(None)
    sentiment_batch_size: int = Field(
        16, ge=1, le=128, description="Batch size for sentiment inference"
    )
    lexicon_file: str | None = Field(None)
    lexicon_weight: float = Field(0.0, ge=0.0, le=1.0)

    @field_validator("directory")
    @classmethod
    def directory_must_exist(cls, v: str) -> str:
        if not os.path.isdir(v):
            raise ValueError(f"Directory not found on server: {v!r}")
        return v

    @field_validator("operation")
    @classmethod
    def operation_must_be_valid(cls, v: str) -> str:
        if v not in {"transcribe", "analyze_conversation"}:
            raise ValueError(f"operation must be 'transcribe' or 'analyze_conversation', got {v!r}")
        return v


class ScanItem(BaseModel):
    file: str
    ok: bool
    error: str | None = None
    data: dict[str, Any] | None = None
    batch_index: int


class ScanProcessResponse(BaseModel):
    items: list[ScanItem]
    ok: int
    failed: int
    total: int
    skipped: int
    timestamp: str
