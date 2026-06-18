"""Pydantic request and response models for the Swedish Sentiment API.

All schemas include field-level validation where it adds value (non-empty lists,
file/directory existence, device string format, etc.).
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from .path_validation import validate_audio_path, validate_directory_path

_AGENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

MAX_FAS4_CALLS = 50
MAX_SEGMENTS_PER_CALL = 200


class AsrParamsMixin(BaseModel):
    """Shared ASR parameters used across transcription-related endpoints."""

    model: str = Field("kb-whisper-large")
    backend: str = Field(
        "faster", description="faster | transformers | whisperx (alignment + diarization)"
    )
    device: str = Field("auto")
    language: str = Field("sv")
    beam_size: int = Field(5, ge=1, le=10)
    vad: bool = Field(True)
    chunk_length_s: int = Field(30, ge=5, le=60)
    revision: str | None = Field(None, description="KB-Whisper revision: standard|strict|subtitle")
    diarize: bool = Field(False, description="Run speaker diarization")
    num_speakers: int | None = Field(None, description="Expected number of speakers (None=auto)")
    hotwords: list[str] | None = Field(None, description="Domain-specific words to boost (callcenter terms etc.)")
    initial_prompt: str | None = Field(None, description="Conditioning prompt for ASR decoder")


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


class TranscribeRequest(AsrParamsMixin):
    audio_path: str = Field(..., description="Path to audio file accessible by the server")
    word_timestamps: bool = Field(True)
    preprocess: bool = Field(False, description="Enable audio preprocessing before ASR")

    @field_validator("audio_path")
    @classmethod
    def audio_path_must_exist(cls, v: str) -> str:
        return validate_audio_path(v)


class TranscribeResponse(BaseModel):
    transcript: dict[str, Any]
    timestamp: str


class TranscribeJobStatus(BaseModel):
    job_id: str
    kind: str
    status: str
    created_at: str
    cancelled: bool = False
    meta: dict[str, Any] = Field(default_factory=dict)


class TranscribeJobListResponse(BaseModel):
    jobs: list[dict[str, Any]]
    timestamp: str


class TranscribeJobCancelResponse(BaseModel):
    job_id: str
    cancelled: bool
    timestamp: str


# ---------------------------------------------------------------------------
# /analyze_conversation
# ---------------------------------------------------------------------------


class AnalyzeConversationRequest(AsrParamsMixin):
    audio_path: str = Field(..., description="Path to audio file accessible by the server")
    word_timestamps: bool = Field(False)
    return_all_scores: bool = Field(True)
    use_full_pipeline: bool = Field(
        False,
        description="Use CallAnalysisPipeline (PII, QA, agent metrics) instead of light transcribe+sentiment path",
    )
    sentiment_profile: str = Field(
        "callcenter",
        description="Sentiment profile for light path (call, callcenter, default, ...)",
    )
    sentiment_model: str | None = Field(None, description="Optional override for sentiment model")
    lexicon_file: str | None = Field(None)
    lexicon_weight: float = Field(0.0, ge=0.0, le=1.0)

    @field_validator("audio_path")
    @classmethod
    def audio_path_must_exist(cls, v: str) -> str:
        return validate_audio_path(v)


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
    pipeline_results: dict[str, Any] | None = Field(
        None,
        description="Full analyzer output when use_full_pipeline=True (agent_performance, qa, pii_redaction, ...)",
    )


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
    # LLM deep analysis (Fas 3)
    use_mistral_llm: bool = Field(False, description="Enable Mistral/OpenRouter holistic analysis")
    llm_model: str | None = Field(None, description="Override Mistral model slug on OpenRouter")
    deep_analysis: bool = Field(False, description="Force deep LLM path")
    llm_api_key: str | None = Field(
        None,
        description="Optional explicit OpenRouter API key (overrides env/file). Use with care over HTTP.",
    )

    @field_validator("segments")
    @classmethod
    def segments_must_not_be_empty(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not v:
            raise ValueError("segments must not be empty")
        if len(v) > MAX_SEGMENTS_PER_CALL:
            raise ValueError(f"segments must have at most {MAX_SEGMENTS_PER_CALL} items")
        return v


class PipelineResponse(BaseModel):
    """Response from the full call analysis pipeline.

    Fas 4 additions: `results` contains the full analyzer output dict (including
    "agent_performance", "qa"/"compliance_qa", "agent_assessment", "customer_metrics",
    "agent_assessment_local" etc.). This makes the new call-center features
    available over the API as required by the plan.
    """

    sentiment_results: list[dict[str, Any]]
    intent_results: list[dict[str, Any]]
    summary: dict[str, Any]
    topics: dict[str, Any]
    insights: dict[str, Any]
    risks: dict[str, Any]
    processing_time_s: float
    timestamp: str
    llm: dict[str, Any] = Field(default_factory=dict, description="Mistral/OpenRouter holistic analysis (when --use-mistral-llm or deep path enabled)")
    results: dict[str, Any] = Field(
        default_factory=dict,
        description="Complete analyzer results (Fas4: agent_performance, qa, agent_assessment, customer_metrics, ...). Use this for new call center features.",
    )


# ---------------------------------------------------------------------------
# /batch_transcribe
# ---------------------------------------------------------------------------


class BatchTranscribeRequest(AsrParamsMixin):
    audio_paths: list[str] | None = None
    directory: str | None = None
    glob: str | None = Field(None, description="Glob pattern within directory, e.g. **/*.wav")
    recursive: bool = True
    limit: int | None = Field(None, ge=1)
    workers: int = Field(1, ge=1, le=8)
    worker_timeout: float = Field(300.0, gt=0.0, description="Per-file worker timeout in seconds")
    word_timestamps: bool = Field(True)


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


class BatchAnalyzeConversationRequest(AsrParamsMixin):
    audio_paths: list[str] | None = None
    directory: str | None = None
    glob: str | None = Field(None)
    recursive: bool = True
    limit: int | None = Field(None, ge=1)
    workers: int = Field(1, ge=1, le=8)
    worker_timeout: float = Field(300.0, gt=0.0, description="Per-file worker timeout in seconds")
    word_timestamps: bool = Field(False)
    # Sentiment
    sentiment_profile: str = Field("callcenter", description="Sentiment profile for light path")
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


class ScanProcessRequest(AsrParamsMixin):
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
    word_timestamps: bool = Field(False)
    use_full_pipeline: bool = Field(
        False,
        description="When operation=analyze_conversation, use full CallAnalysisPipeline per file",
    )
    # Sentiment (used when operation=analyze_conversation)
    sentiment_profile: str = Field("callcenter", description="Sentiment profile for light analyze path")
    sentiment_model: str | None = Field(None)
    sentiment_batch_size: int = Field(
        16, ge=1, le=128, description="Batch size for sentiment inference"
    )
    lexicon_file: str | None = Field(None)
    lexicon_weight: float = Field(0.0, ge=0.0, le=1.0)

    @field_validator("directory")
    @classmethod
    def directory_must_exist(cls, v: str) -> str:
        return validate_directory_path(v)

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


# ---------------------------------------------------------------------------
# Fas 4.5.2: New endpoints for call center features (agent perf, search, insights, qa, alerts)
# These use the extended pipeline methods (cached aggregates, semantic search, etc.)
# ---------------------------------------------------------------------------


def _validate_fas4_segments_list(v: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    if not v:
        raise ValueError("segments_list must not be empty")
    if len(v) > MAX_FAS4_CALLS:
        raise ValueError(f"segments_list must have at most {MAX_FAS4_CALLS} calls")
    for i, call_segs in enumerate(v):
        if not call_segs:
            raise ValueError(f"segments_list[{i}] must not be empty")
        if len(call_segs) > MAX_SEGMENTS_PER_CALL:
            raise ValueError(
                f"segments_list[{i}] must have at most {MAX_SEGMENTS_PER_CALL} segments"
            )
    return v


class Fas4LlmFlags(BaseModel):
    """Shared LLM flags for Fas 4 pipeline endpoints."""

    reanalyze: bool = Field(
        False,
        description="Force re-analysis of all calls; default uses per-call report cache",
    )
    use_mistral_llm: bool = Field(False, description="Enable Mistral/OpenRouter holistic analysis")
    llm_model: str | None = Field(None, description="Override Mistral model slug on OpenRouter")
    deep_analysis: bool = Field(False, description="Force deep LLM path")
    llm_api_key: str | None = Field(
        None,
        description="Deprecated: prefer X-OpenRouter-Key header. Requires API_ALLOW_CLIENT_LLM_KEY.",
    )


class AgentPerformanceRequest(Fas4LlmFlags):
    """Request for /agent_performance endpoint. Provide segments for one or more calls."""
    segments_list: list[list[dict[str, Any]]] = Field(..., description="List of segment lists (one per call)")
    agent_id: str = Field(..., description="Agent identifier to aggregate for")
    window: str = Field("7d", description="Time window e.g. 7d, 30d")
    profile: str = Field("callcenter")

    @field_validator("segments_list")
    @classmethod
    def validate_segments_list(cls, v: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
        return _validate_fas4_segments_list(v)

    @field_validator("agent_id")
    @classmethod
    def agent_id_format(cls, v: str) -> str:
        if not _AGENT_ID_RE.match(v):
            raise ValueError(
                "agent_id must be 1-64 chars: alphanumeric start, then letters, digits, . _ -"
            )
        return v

    @model_validator(mode="after")
    def path_agent_matches_body(self) -> AgentPerformanceRequest:
        # Path param validated in router; body agent_id must match when both present
        return self


class AgentPerformanceResponse(BaseModel):
    agent_id: str
    metrics: dict[str, Any]
    cached: bool = False
    timestamp: str


class SemanticSearchRequest(Fas4LlmFlags):
    segments_list: list[list[dict[str, Any]]] = Field(..., description="List of calls to index/search over")
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(5, ge=1, le=50)
    filters: dict[str, Any] | None = Field(None)
    profile: str = Field("callcenter")

    @field_validator("segments_list")
    @classmethod
    def validate_segments_list(cls, v: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
        return _validate_fas4_segments_list(v)


class SemanticSearchResponse(BaseModel):
    query: str
    hits: list[dict[str, Any]]
    meta: dict[str, Any]
    timestamp: str


class HotTopicsRequest(Fas4LlmFlags):
    segments_list: list[list[dict[str, Any]]]
    window: str = "7d"
    profile: str = "callcenter"

    @field_validator("segments_list")
    @classmethod
    def validate_segments_list(cls, v: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
        return _validate_fas4_segments_list(v)


class HotTopicsResponse(BaseModel):
    hot_topics: list[dict[str, Any]]
    meta: dict[str, Any]
    timestamp: str


class QAScoreRequest(Fas4LlmFlags):
    segments: list[dict[str, Any]]
    profile: str = "callcenter"

    @field_validator("segments")
    @classmethod
    def segments_must_not_be_empty(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not v:
            raise ValueError("segments must not be empty")
        if len(v) > MAX_SEGMENTS_PER_CALL:
            raise ValueError(f"segments must have at most {MAX_SEGMENTS_PER_CALL} items")
        return v


class QAScoreResponse(BaseModel):
    qa: dict[str, Any]
    timestamp: str


class AlertsRequest(Fas4LlmFlags):
    segments_list: list[list[dict[str, Any]]] | None = None  # for per call
    aggregate: dict[str, Any] | None = None  # for trend alerts from aggregator
    profile: str = "callcenter"

    @model_validator(mode="after")
    def require_input(self) -> AlertsRequest:
        if not self.segments_list and not self.aggregate:
            raise ValueError("Either segments_list or aggregate must be provided")
        if self.segments_list:
            _validate_fas4_segments_list(self.segments_list)
        return self


class AlertsResponse(BaseModel):
    alerts: list[dict[str, Any]]
    timestamp: str
