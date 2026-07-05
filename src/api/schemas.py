"""Pydantic request and response models for the Swedish Sentiment API.

All schemas include field-level validation where it adds value (non-empty lists,
file/directory existence, device string format, etc.).
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from .path_validation import (
    validate_audio_path,
    validate_batch_audio_input,
    validate_directory_path,
    validate_lexicon_path,
    validate_state_file_path,
)

_AGENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

MAX_FAS4_CALLS = 50
MAX_SEGMENTS_PER_CALL = 200
MAX_ANALYZE_TEXTS = 1000


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
    hotwords: list[str] | None = Field(
        None, description="Domain-specific words to boost (callcenter terms etc.)"
    )
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
        if len(v) > MAX_ANALYZE_TEXTS:
            raise ValueError(f"texts must not exceed {MAX_ANALYZE_TEXTS} items")
        return v

    @field_validator("lexicon_file")
    @classmethod
    def lexicon_file_must_be_allowed(cls, v: str | None) -> str | None:
        if v is None:
            return v
        return validate_lexicon_path(v)


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
    preprocess_mode: str | None = Field(
        None,
        description="Preprocess mode: off | basic | callcenter (v2 bandpass + tuned VAD). Overrides legacy boolean when set.",
    )

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

    @field_validator("lexicon_file")
    @classmethod
    def lexicon_file_must_be_allowed(cls, v: str | None) -> str | None:
        if v is None:
            return v
        return validate_lexicon_path(v)


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
    profile: str = Field(
        "default",
        description="Analysis profile (callcenter, sales, complaint, support, teknisk_support, ...)",
    )
    selected_analyzers: list[str] | None = Field(
        None,
        description="Explicit analyzer subset (overrides profile default_selected; deps auto-resolved)",
    )
    async_analyzers: bool = Field(
        False,
        description="Run independent analyzers in parallel within dependency levels",
    )
    sentiment_model: str | None = Field(None, description="Optional sentiment model override")
    device: str = Field("auto")
    # LLM deep analysis (Fas 3)
    use_mistral_llm: bool = Field(False, description="Enable LLM holistic analysis")
    llm_model: str | None = Field(None, description="Override LLM model slug")
    deep_analysis: bool = Field(False, description="Force deep LLM path")
    llm_api_key: str | None = Field(
        None,
        description="Deprecated: prefer X-OpenRouter-Key header. Requires API_ALLOW_CLIENT_LLM_KEY.",
    )
    provider: str = Field(
        "openrouter",
        description="LLM provider: openrouter (default) | groq",
        pattern=r"^(openrouter|groq)$",
    )
    groq_eu_residency: bool = Field(
        False,
        description="GDPR gate for Groq: affirm EU data residency (default: OFF).",
    )

    @field_validator("segments")
    @classmethod
    def segments_must_not_be_empty(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not v:
            raise ValueError("segments must not be empty")
        if len(v) > MAX_SEGMENTS_PER_CALL:
            raise ValueError(f"segments must have at most {MAX_SEGMENTS_PER_CALL} items")
        return v


# ---------------------------------------------------------------------------
# Typed analyzer output models (Fas 5 — full analyzer surface in API/UI)
# These mirror the dict shapes returned by each analyzer in src/analysis/.
# `PipelineResponse.results` stays `dict[str, Any]` for backward compatibility,
# but these models document the expected shape and can be used for validation.
# ---------------------------------------------------------------------------


class EmotionSegmentResult(BaseModel):
    """Per-segment emotion output from the `emotion` analyzer."""

    primary: str = Field(..., description="Primary emotion label (frustration, ilska, besvikelse, förvirring, tillfredsställelse, neutral, oro, glädje)")
    scores: dict[str, float] = Field(default_factory=dict, description="Score per emotion label")
    speaker: str | None = None


class AspectItem(BaseModel):
    """One aspect-based sentiment item from the `aspect` analyzer."""

    aspect: str
    sentiment: str = Field(..., description="positive | negative | neutral")
    score: float = 0.0
    evidence: str | None = None
    start: float | None = None
    end: float | None = None
    speaker: str | None = None


class TrajectoryResult(BaseModel):
    """Output from the `trajectory` analyzer."""

    customer_sentiment_slope: float = 0.0
    escalation_events: int = 0
    escalation_event_details: list[dict[str, Any]] = Field(default_factory=list)
    peak_frustration_turn: int | None = None
    sentiment_trend: list[float] = Field(default_factory=list)


class RootCauseItem(BaseModel):
    cause: str
    count: int = 1
    recommendation: str | None = None


class RootCauseResult(BaseModel):
    """Output from the `root_cause` analyzer."""

    root_causes: list[RootCauseItem] = Field(default_factory=list)
    top_root_cause: str | None = None
    evidence_examples: list[dict[str, Any]] = Field(default_factory=list)
    overall_risk: str = Field("low", description="low | medium | high")
    message: str | None = None


class CoachingInsightItem(BaseModel):
    rule_id: str
    priority: str = Field(..., description="high | medium | low")
    recommendation: str
    evidence: Any = None


class CoachingResult(BaseModel):
    """Output from the `actionable_coaching` analyzer."""

    coaching_insights: list[CoachingInsightItem] = Field(default_factory=list)
    top_recommendation: str | None = None
    insight_count: int = 0


class CustomerEffortSegment(BaseModel):
    speaker: str | None = None
    start: float = 0.0
    end: float = 0.0
    effort_score: float = 0.0


class CustomerEffortResult(BaseModel):
    """Output from the `customer_effort` analyzer."""

    overall_ces: float = 0.0
    scale: str = "0-100 (högre = mer effort/frustration)"
    per_segment: list[CustomerEffortSegment] = Field(default_factory=list)
    coaching_tips: list[str] = Field(default_factory=list)


class ActiveListeningResult(BaseModel):
    """Output from the `active_listening` analyzer."""

    listening_score: float = 50.0
    backchannel_count: int = 0
    speaker_balance: dict[str, float] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)
    tips: list[str] = Field(default_factory=list)


class EmpathySegment(BaseModel):
    speaker: str | None = None
    start: float | None = None
    empathy_score: float = 0.0
    evidence: list[str] = Field(default_factory=list)


class EmpathyResult(BaseModel):
    """Output from the `empathy` analyzer."""

    overall_empathy: float = 50.0
    scale: str = "0-100 (högre = bättre empati)"
    per_segment: list[EmpathySegment] = Field(default_factory=list)
    coaching_tips: list[str] = Field(default_factory=list)


class ResolutionProbabilityResult(BaseModel):
    """Output from the `resolution_probability` analyzer."""

    resolution_probability: float = 50.0
    confidence: int = 30
    recommended_action: str = ""
    factors: dict[str, Any] = Field(default_factory=dict)


class JourneyStage(BaseModel):
    stage: str
    start: float = 0.0
    speaker: str | None = None
    text_snippet: str = ""
    intent: str | None = None
    sentiment: str | None = None


class MultiTurnJourneyResult(BaseModel):
    """Output from the `multi_turn_journey` analyzer."""

    journey_stages: list[JourneyStage] = Field(default_factory=list)
    resolved: bool = False
    unresolved_count: int = 0
    key_turning_points: list[dict[str, Any]] = Field(default_factory=list)
    recommendation: str | None = None
    message: str | None = None


class UpsellOpportunity(BaseModel):
    speaker: str | None = None
    start: float = 0.0
    end: float = 0.0
    confidence: float = 0.0
    signals: list[str] = Field(default_factory=list)
    suggested_action: str = ""
    evidence: str = ""


class UpsellResult(BaseModel):
    """Output from the `upsell_opportunity` analyzer."""

    opportunities: list[UpsellOpportunity] = Field(default_factory=list)
    count: int = 0
    recommendation: str | None = None


class DialectSensitivityResult(BaseModel):
    """Output from the `dialect_sensitivity` analyzer."""

    dialect_risk_level: str = Field("low", description="low | medium | high")
    flagged_segments: list[dict[str, Any]] = Field(default_factory=list)
    total_dialect_hits: int = 0
    slang_count: int = 0
    recommendation: str | None = None


class ComplianceRiskResult(BaseModel):
    """Output from the `compliance_risk` analyzer."""

    overall_risk_level: str = Field("low", description="low | medium | high")
    flagged_segments: list[dict[str, Any]] = Field(default_factory=list)
    recommendation: str | None = None


class RoleClassifierResult(BaseModel):
    """Output from the `role` analyzer."""

    roles: dict[str, str] = Field(default_factory=dict, description="speaker -> agent|customer")
    talk_ratios: dict[str, float] = Field(default_factory=dict)
    question_density: dict[str, float] = Field(default_factory=dict)
    lexical_formality: float = 0.0
    intervention_count: int = 0
    sentiment_variance: float = 0.0
    num_agent_turns: int = 0
    num_customer_turns: int = 0


class PredictiveResult(BaseModel):
    """Output from the `predictive` analyzer (RiskAnalyzer)."""

    churn_risk: float = 0.0
    escalation_risk: float = 0.0
    satisfaction_score: float = 0.5
    risk_factors: list[str] = Field(default_factory=list)
    risk_level: str = Field("low", description="low | medium | high | critical")
    recommended_action: str | None = None


class AnalyzerResults(BaseModel):
    """Typed view of `PipelineResponse.results`.

    All fields are optional because analyzers may be skipped (profile config,
    LLM deep path, or graceful degradation). Consumers should null-check each
    field before rendering.
    """

    emotion: list[EmotionSegmentResult] | None = None
    aspect: list[AspectItem] | None = None
    trajectory: TrajectoryResult | None = None
    root_cause: RootCauseResult | None = None
    actionable_coaching: CoachingResult | None = None
    customer_effort: CustomerEffortResult | None = None
    active_listening: ActiveListeningResult | None = None
    empathy: EmpathyResult | None = None
    resolution_probability: ResolutionProbabilityResult | None = None
    multi_turn_journey: MultiTurnJourneyResult | None = None
    upsell_opportunity: UpsellResult | None = None
    dialect_sensitivity: DialectSensitivityResult | None = None
    compliance_risk: ComplianceRiskResult | None = None
    role: RoleClassifierResult | None = None
    predictive: PredictiveResult | None = None
    # Fas 4 enrichment (already present but typed here for documentation)
    agent_performance: dict[str, Any] | None = None
    qa: dict[str, Any] | None = None
    agent_assessment: dict[str, Any] | None = None
    agent_assessment_local: dict[str, Any] | None = None
    customer_metrics: dict[str, Any] | None = None
    pii_redaction: dict[str, Any] | None = None
    alerts: list[dict[str, Any]] | None = None
    llm_judge: dict[str, Any] | None = None


class PipelineResponse(BaseModel):
    """Response from the full call analysis pipeline.

    Fas 4 additions: `results` contains the full analyzer output dict (including
    "agent_performance", "qa"/"compliance_qa", "agent_assessment", "customer_metrics",
    "agent_assessment_local" etc.). This makes the new call-center features
    available over the API as required by the plan.

    Fas 5: `analyzer_results` provides a typed view of the same data for
    documentation and type-safe consumers. `results` stays `dict[str, Any]`
    for backward compatibility.
    """

    sentiment_results: list[dict[str, Any]]
    intent_results: list[dict[str, Any]]
    summary: dict[str, Any]
    topics: dict[str, Any]
    insights: dict[str, Any]
    risks: dict[str, Any]
    processing_time_s: float
    timestamp: str
    llm: dict[str, Any] = Field(
        default_factory=dict,
        description="Mistral/OpenRouter holistic analysis (when --use-mistral-llm or deep path enabled)",
    )
    results: dict[str, Any] = Field(
        default_factory=dict,
        description="Complete analyzer results (Fas4: agent_performance, qa, agent_assessment, customer_metrics, ...). Use this for new call center features.",
    )
    analyzer_results: AnalyzerResults | None = Field(
        None,
        description="Typed view of `results` (Fas 5). Null when no analyzers ran or for backward-compatible clients.",
    )


def _safe_parse(model_cls: type[BaseModel], data: Any) -> Any | None:
    """Best-effort parse of an analyzer output dict into a typed model.

    Returns None on failure so that one malformed analyzer output does not
    break the whole response.
    """
    if data is None:
        return None
    try:
        return model_cls.model_validate(data)
    except Exception:  # noqa: BLE001 — graceful degradation for typed view
        return None


def build_analyzer_results(results: dict[str, Any]) -> AnalyzerResults:
    """Build a typed `AnalyzerResults` from the raw `report.results` dict.

    Each analyzer field is parsed best-effort; unparseable fields are left as
    None so the frontend can gracefully skip them.
    """
    if not results:
        return AnalyzerResults()

    # emotion + aspect are lists
    emotion_raw = results.get("emotion")
    emotion: list[EmotionSegmentResult] | None = None
    if isinstance(emotion_raw, list):
        emotion = [
            _safe_parse(EmotionSegmentResult, e) for e in emotion_raw if isinstance(e, dict)
        ]
        emotion = [e for e in emotion if e is not None] or None

    aspect_raw = results.get("aspect")
    aspect: list[AspectItem] | None = None
    if isinstance(aspect_raw, list):
        aspect = [
            _safe_parse(AspectItem, a) for a in aspect_raw if isinstance(a, dict)
        ]
        aspect = [a for a in aspect if a is not None] or None

    return AnalyzerResults(
        emotion=emotion,
        aspect=aspect,
        trajectory=_safe_parse(TrajectoryResult, results.get("trajectory")),
        root_cause=_safe_parse(RootCauseResult, results.get("root_cause")),
        actionable_coaching=_safe_parse(CoachingResult, results.get("actionable_coaching")),
        customer_effort=_safe_parse(CustomerEffortResult, results.get("customer_effort")),
        active_listening=_safe_parse(ActiveListeningResult, results.get("active_listening")),
        empathy=_safe_parse(EmpathyResult, results.get("empathy")),
        resolution_probability=_safe_parse(
            ResolutionProbabilityResult, results.get("resolution_probability")
        ),
        multi_turn_journey=_safe_parse(
            MultiTurnJourneyResult, results.get("multi_turn_journey")
        ),
        upsell_opportunity=_safe_parse(UpsellResult, results.get("upsell_opportunity")),
        dialect_sensitivity=_safe_parse(
            DialectSensitivityResult, results.get("dialect_sensitivity")
        ),
        compliance_risk=_safe_parse(ComplianceRiskResult, results.get("compliance_risk")),
        role=_safe_parse(RoleClassifierResult, results.get("role")),
        predictive=_safe_parse(PredictiveResult, results.get("predictive")),
        # Fas 4 enrichment — kept as dict for forward compatibility
        agent_performance=results.get("agent_performance"),
        qa=results.get("qa"),
        agent_assessment=results.get("agent_assessment"),
        agent_assessment_local=results.get("agent_assessment_local"),
        customer_metrics=results.get("customer_metrics"),
        pii_redaction=results.get("pii_redaction"),
        alerts=results.get("alerts"),
        llm_judge=results.get("llm_judge"),
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

    @field_validator("directory")
    @classmethod
    def directory_must_exist(cls, v: str | None) -> str | None:
        if v is None:
            return v
        return validate_directory_path(v)

    @field_validator("audio_paths")
    @classmethod
    def audio_paths_must_be_allowed(cls, v: list[str] | None) -> list[str] | None:
        if not v:
            return v
        return [validate_batch_audio_input(p) for p in v]

    @model_validator(mode="after")
    def require_audio_source(self) -> BatchTranscribeRequest:
        if not self.audio_paths and not self.directory:
            raise ValueError("Either audio_paths or directory must be provided")
        return self


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
    sentiment_batch_size: int = Field(
        16, ge=1, le=128, description="Batch size for sentiment inference"
    )
    lexicon_file: str | None = Field(None)
    lexicon_weight: float = Field(0.0, ge=0.0, le=1.0)

    @field_validator("directory")
    @classmethod
    def directory_must_exist(cls, v: str | None) -> str | None:
        if v is None:
            return v
        return validate_directory_path(v)

    @field_validator("audio_paths")
    @classmethod
    def audio_paths_must_be_allowed(cls, v: list[str] | None) -> list[str] | None:
        if not v:
            return v
        return [validate_batch_audio_input(p) for p in v]

    @field_validator("lexicon_file")
    @classmethod
    def lexicon_file_must_be_allowed(cls, v: str | None) -> str | None:
        if v is None:
            return v
        return validate_lexicon_path(v)

    @model_validator(mode="after")
    def require_audio_source(self) -> BatchAnalyzeConversationRequest:
        if not self.audio_paths and not self.directory:
            raise ValueError("Either audio_paths or directory must be provided")
        return self


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
    sentiment_profile: str = Field(
        "callcenter", description="Sentiment profile for light analyze path"
    )
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

    @field_validator("state_file")
    @classmethod
    def state_file_must_be_allowed(cls, v: str | None) -> str | None:
        if v is None:
            return v
        return validate_state_file_path(v)

    @field_validator("lexicon_file")
    @classmethod
    def lexicon_file_must_be_allowed(cls, v: str | None) -> str | None:
        if v is None:
            return v
        return validate_lexicon_path(v)

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
    use_mistral_llm: bool = Field(False, description="Enable LLM holistic analysis")
    llm_model: str | None = Field(None, description="Override LLM model slug")
    deep_analysis: bool = Field(False, description="Force deep LLM path")
    llm_api_key: str | None = Field(
        None,
        description="Deprecated: prefer X-OpenRouter-Key header. Requires API_ALLOW_CLIENT_LLM_KEY.",
    )
    provider: str = Field(
        "openrouter",
        description="LLM provider: openrouter (default) | groq",
        pattern=r"^(openrouter|groq)$",
    )
    groq_eu_residency: bool = Field(
        False,
        description="GDPR gate for Groq: affirm EU data residency (default: OFF).",
    )


class AgentPerformanceRequest(Fas4LlmFlags):
    """Request for /agent_performance endpoint. Provide segments for one or more calls."""

    segments_list: list[list[dict[str, Any]]] = Field(
        ..., description="List of segment lists (one per call)"
    )
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
    segments_list: list[list[dict[str, Any]]] = Field(
        ..., description="List of calls to index/search over"
    )
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
