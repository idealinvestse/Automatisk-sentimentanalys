"""Pydantic models for structured Mistral/OpenRouter LLM output (Task 3.1.3).

These models define the exact shape of the holistic post-transcription analysis
returned by ConversationMistralAnalyzer (Task 3.1.2).

Design choices & rationale:
- Pydantic v2 BaseModel with Field(..., description=...) for rich documentation that
  also becomes part of the JSON schema sent to OpenRouter (`json_schema` + strict:true).
  This gives the model (Mistral Medium 3.5 / Large 3) very strong guidance on what
  "good" evidence-based Swedish callcenter analysis looks like.
- Strict mode friendly: model_config extra='forbid', no default factories that break
  schema generation, explicit required fields where the LLM must reason.
- Separate small models (AspectItem, RootCause, AgentAssessment, etc.) so the analyzer
  can request task subsets later and so that merging into CallAnalysisReport is granular.
- All textual fields emphasize "evidensspann" (evidence spans) and "reasoning" – this
  is key for callcenter use case: QA teams need to be able to audit *why* the LLM
  reached a conclusion on a real Swedish conversation (compliance, coaching).
- Swedish-first: descriptions and examples are in Swedish where they reflect output
  language. The LLM is instructed (in the analyzer prompt) to produce Swedish text
  for summaries/insights while keeping keys stable (English for code).
- Validation: after LLM returns JSON we do `CallLLMOutput.model_validate(...)` so any
  drift is caught early (instead of silent bad data in reports).
- European integrity: The schema itself documents that this data may have been
  produced by sending PII-laden transcripts to OpenRouter/Mistral. Consumers of the
  report (dashboard, API) can surface "LLM-enhanced (external)" badges.

These models are the contract between the remote LLM and our local hybrid pipeline.
They are intentionally additive – the local fast path (sentiment, trajectory heuristics,
role, insights) remains the base; Mistral only enriches/overrides complex fields.

See UTVECKLINGSPLAN_Mistral_OpenRouter_LLM_Integration.md Task 3.1.3 and bilaga for
the expected top-level shape.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EvidenceSpan(BaseModel):
    """A concrete span of text (with optional speaker) used as evidence for a claim."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., description="The exact quote or close paraphrase from the transcript.")
    speaker_role: str | None = Field(
        None, description="agent | customer | unknown (if known from role inference)"
    )
    turn_index: int | None = Field(None, description="0-based turn number in the conversation.")


class EmotionTrajectoryPoint(BaseModel):
    """Single point on the emotion/sentiment trajectory over the conversation."""

    model_config = ConfigDict(extra="forbid")

    turn: int = Field(..., ge=0, description="Turn index (customer or overall turns).")
    sentiment: float = Field(..., ge=-1.0, le=1.0, description="Aggregated sentiment score at this turn (-1=very negative ... +1).")
    primary_emotion: str | None = Field(None, description="Dominant emotion label at this turn (frustration, ilska, etc).")
    score: float = Field(0.0, ge=0.0, le=1.0, description="Confidence or intensity of the primary emotion.")


class Trajectory(BaseModel):
    """Holistic view of how the conversation developed (customer-focused for callcenter)."""

    model_config = ConfigDict(extra="forbid")

    points: list[EmotionTrajectoryPoint] = Field(
        default_factory=list, description="Time series of sentiment/emotion over turns."
    )
    customer_sentiment_slope: float = Field(
        0.0, description="Overall slope of customer sentiment (negative = deteriorating)."
    )
    escalation_events: list[str] = Field(
        default_factory=list,
        description="Descriptions of moments where frustration/escalation spiked (with evidence).",
    )
    summary: str = Field(
        ..., description="Concise Swedish narrative of the emotional arc and key turning points."
    )


class AspectItem(BaseModel):
    """A refined aspect (topic + sentiment) discovered or improved by holistic reasoning."""

    model_config = ConfigDict(extra="forbid")

    aspect: str = Field(
        ..., description="One of the callcenter aspects or a discovered sub-aspect, e.g. fakturering_pris, agent_attityd."
    )
    sentiment: str = Field(..., description="negativ | neutral | positiv (after full context).")
    score: float = Field(0.7, ge=0.0, le=1.0)
    evidence: list[EvidenceSpan] = Field(
        default_factory=list, description="Concrete quotes that support this aspect judgment."
    )
    related_to: list[str] = Field(
        default_factory=list, description="Other aspects this one is causally or emotionally linked to."
    )


class RootCause(BaseModel):
    """Root cause analysis – the real underlying problem, not just the surface complaint."""

    model_config = ConfigDict(extra="forbid")

    primary_cause: str = Field(..., description="Short Swedish description of the root cause.")
    contributing_factors: list[str] = Field(
        default_factory=list, description="Other factors that made the problem worse."
    )
    evidence_spans: list[EvidenceSpan] = Field(
        default_factory=list, description="Key transcript parts that prove this is the root."
    )
    customer_unresolved: bool = Field(
        True, description="True if the customer's core need was not resolved by end of call."
    )


class ActionableSummary(BaseModel):
    """Structured, coachable output for QA / team leads."""

    model_config = ConfigDict(extra="forbid")

    problem: str = Field(..., description="What the customer actually experienced (Swedish).")
    resolution_attempts: list[str] = Field(
        default_factory=list, description="What the agent tried (or failed to try)."
    )
    final_customer_state: str = Field(
        ..., description="How the customer felt/left the conversation (frustrated, relieved, etc)."
    )
    recommendations_for_qa: list[str] = Field(
        default_factory=list,
        description="Concrete, actionable coaching points for the agent or process (e.g. 'Använd empatifraser tidigare').",
    )
    risk_level: str = Field("medium", description="low | medium | high (escalation/compliance risk).")


class AgentAssessment(BaseModel):
    """Assessment of the agent's performance with evidence (key for callcenter KPIs).

    Enhanced in Fas 4.1: now includes structured coaching recs with evidence_spans (from LLM in 4.1.2)
    and is merged with local quantitative metrics from agent_performance engine.
    """

    model_config = ConfigDict(extra="forbid")

    empathy_score: float = Field(
        ..., ge=0.0, le=1.0, description="How well the agent showed understanding and de-escalated (0-1)."
    )
    compliance_flags: list[str] = Field(
        default_factory=list, description="Script/process violations or missed opportunities detected."
    )
    strengths: list[str] = Field(default_factory=list, description="Positive behaviours worth reinforcing.")
    weaknesses: list[str] = Field(
        default_factory=list, description="Areas for improvement (specific, not generic)."
    )
    evidence_spans: list[EvidenceSpan] = Field(
        default_factory=list, description="Exact turns/phrases that justify the empathy score and flags."
    )
    specific_coaching_recommendations: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Actionable, evidence-based coaching items. Each item: "
            "{'recommendation': str, 'evidence_spans': list[EvidenceSpan], 'priority': 'high'|'medium'|'low', 'category': str}. "
            "Produced by Mistral (4.1.2) or hybrid rules."
        ),
    )
    overall_assessment: str | None = Field(
        None, description="Concise Swedish overall judgment suitable for coach review."
    )


class CallLLMOutput(BaseModel):
    """Top-level structured response from Mistral for a full callcenter conversation.

    This is what gets validated after the LLM returns JSON and what is later merged
    into CallAnalysisReport.results["llm"] (or similar) in the hybrid pipeline.
    """

    model_config = ConfigDict(extra="forbid")

    trajectory: Trajectory | None = Field(
        None, description="Full conversation trajectory + escalation narrative."
    )
    refined_aspects: list[AspectItem] = Field(
        default_factory=list, description="Aspects with cross-turn context and evidence."
    )
    root_cause: RootCause | None = Field(None, description="Deep root cause + unresolved status.")
    actionable_summary: ActionableSummary | None = Field(
        None, description="QA-ready problem + recommendations."
    )
    agent_assessment: AgentAssessment | None = Field(
        None, description="Agent performance with auditable evidence."
    )
    emotion_trajectory: list[EmotionTrajectoryPoint] = Field(
        default_factory=list, description="Granular emotion points (can feed plots)."
    )

    meta: dict[str, Any] = Field(
        default_factory=dict,
        description="Model used, tokens, cost_usd, latency, cached flag, etc. (populated by client).",
    )

    # Convenience: the raw model name for quick checks
    @property
    def model_used(self) -> str | None:
        return self.meta.get("model")


# For convenience when building the json_schema for the client
LLM_OUTPUT_JSON_SCHEMA = CallLLMOutput.model_json_schema()


# =============================================================================
# Fas 4.1+ : Agent / Customer metrics (local + hybrid) - Pydantic for merging into CallAnalysisReport
# =============================================================================

class AgentMetrics(BaseModel):
    """Quantitative agent performance signals derived locally (Fas 4.1.1).

    Always cheap/fast to compute. Used for:
    - Per-call dashboard cards
    - Agent benchmarking / trends over time (aggregated in insights_aggregator later)
    - Input features to LLM for nuanced coaching (4.1.2)

    All scores 0-1 unless noted. Evidence is indirect via flags + later LLM spans.
    """

    model_config = ConfigDict(extra="forbid")

    talk_ratio: float = Field(..., ge=0.0, le=1.0, description="Agent's share of total speaking time in call (0-1).")
    talk_listen_ratio: float = Field(..., ge=0.0, description="Agent talk time divided by customer talk time (>1 = agent dominates airtime).")
    question_density: float = Field(0.0, ge=0.0, description="Approx questions asked by agent per agent turn.")
    lexical_formality: float = Field(0.5, ge=0.0, le=1.0, description="Heuristic score for professional Swedish service language (higher = more formal/polite).")
    sentiment_variance: float = Field(0.0, ge=0.0, description="Variance in agent's sentiment tone across their turns (high = inconsistent).")
    intervention_count: int = Field(0, ge=0, description="Approx number of times agent 'intervened' (speaker switches into customer flow).")
    empathy_score: float = Field(0.0, ge=0.0, le=1.0, description="Local rule-based empathy signal (0-1). Combined with LLM later.")
    de_escalation_effectiveness: float = Field(0.0, ge=0.0, le=1.0, description="Local measure of recovery in customer sentiment after negative moments.")
    compliance_flags: list[str] = Field(
        default_factory=list, description="Rule-detected issues e.g. 'missing_greeting', 'no_empathy_on_frustration'."
    )
    num_agent_turns: int = Field(0, ge=0)
    num_customer_turns: int = Field(0, ge=0)
    total_talk_time_s: float = Field(0.0, ge=0.0, description="Total duration of all segments (agent+customer) as proxy for call length.")


# =============================================================================
# Fas 4.3: Aggregated insights models (Pydantic, mergable into report or separate aggregate output)
# =============================================================================

class HotTopic(BaseModel):
    """A hot topic aggregated across multiple calls (Fas 4.3.1)."""

    model_config = ConfigDict(extra="forbid")

    topic: str = Field(..., description="Key e.g. 'fakturering_pris', 'leverans', 'agent_attityd'")
    volume: int = Field(..., ge=0, description="Number of calls mentioning this topic")
    avg_sentiment: float = Field(..., ge=-1.0, le=1.0)
    trend: str = Field("stable", description="up | down | stable (based on time or volume change)")
    evidence_spans: list[EvidenceSpan] = Field(
        default_factory=list, description="Concrete quotes from representative calls."
    )
    sample_quotes: list[str] = Field(default_factory=list)
    llm_summary: str | None = Field(
        None, description="Optional nuanced description generated via Mistral (when available)."
    )


class AggregatedInsights(BaseModel):
    """Output from InsightsAggregator. Can be stored separately or attached to batch results."""

    model_config = ConfigDict(extra="forbid")

    hot_topics: list[HotTopic] = Field(default_factory=list, description="Top hot topics with volume, sentiment, trend, evidence")
    sentiment_trends: dict[str, Any] = Field(
        default_factory=dict, description="Overall sentiment slope, buckets over time, per topic trends"
    )
    root_cause_clusters: list[dict[str, Any]] = Field(
        default_factory=list, description="Clustered root causes (with size, representative evidence)"
    )
    top_agent_issues: list[dict[str, Any]] = Field(
        default_factory=list, description="Issues per agent or team (from agent_performance + assessments)"
    )
    meta: dict[str, Any] = Field(
        default_factory=dict,
        description="num_calls, time_window, generated_at, llm_used_for_descriptions, embedding_model etc.",
    )


# =============================================================================
# Fas 4.4.1: PII Redaction Log (structured, evidence-based, for audit + results merge)
# =============================================================================

class PiiRedactionEvent(BaseModel):
    """A single PII redaction event (for logging and audit)."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(..., description="personnummer | email | phone | credit_card | address | name | other")
    original: str = Field(..., description="The original matched text (truncated for log safety).")
    replacement: str = Field(..., description="The token used, e.g. [REDACTED_PNR]")
    segment_index: int | None = Field(None, description="Index in the segment list.")
    char_start: int | None = Field(None)
    char_end: int | None = Field(None)


class PiiRedactionLog(BaseModel):
    """Structured log of all PII redactions performed on a call (early pipeline, Fas 4.4.1).

    Attached to CallAnalysisReport.results["pii_redaction"] when applied.
    Allows QA/audit to know exactly what was masked without exposing the PII.
    """

    model_config = ConfigDict(extra="forbid")

    events: list[PiiRedactionEvent] = Field(default_factory=list)
    total_redacted: int = Field(0, description="Total number of PII items redacted.")
    types_redacted: list[str] = Field(default_factory=list, description="Unique types found (e.g. ['personnummer', 'email']).")
    applied_to_local: bool = Field(True, description="Whether redaction affected local analyzers (sentiment, role, etc) and not only LLM.")
    profile: str = "default"
    timestamp: str = Field(default_factory=lambda: __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat())


# =============================================================================
# Fas 4.4.2: Alerting models (Pydantic, evidence-based, mergable into results["alerts"])
# =============================================================================

class Alert(BaseModel):
    """A triggered alert (Fas 4.4.2). Actionable with evidence."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(..., description="Identifier for the rule that triggered, e.g. 'high_escalation_risk'")
    severity: str = Field("medium", description="low | medium | high | critical")
    message: str = Field(..., description="Human readable Swedish description of the alert.")
    evidence_spans: list[EvidenceSpan] = Field(
        default_factory=list, description="Concrete transcript parts that caused the alert."
    )
    triggered_values: dict[str, Any] = Field(
        default_factory=dict, description="The values that matched the rule, e.g. {'customer_sentiment': -0.8, 'escalation_risk': 0.7}"
    )
    recommended_actions: list[str] = Field(
        default_factory=list,
        description="Actionable next steps, e.g. ['flag_supervisor', 'create_coaching_task:empathy', 'notify_webhook']"
    )
    source: str = Field("per_call", description="per_call | aggregator_trend | batch")


class AlertSummary(BaseModel):
    """Summary of alerts for a call or batch."""

    model_config = ConfigDict(extra="forbid")

    alerts: list[Alert] = Field(default_factory=list)
    highest_severity: str = "none"
    total: int = 0
    meta: dict[str, Any] = Field(default_factory=dict)


class CustomerMetrics(BaseModel):
    """Customer-side signals (symmetric to agent for context in coaching)."""

    model_config = ConfigDict(extra="forbid")

    talk_ratio: float = Field(..., ge=0.0, le=1.0, description="Customer's share of total speaking time.")
    sentiment_slope: float = Field(0.0, description="Rough delta: end sentiment - start sentiment for customer (-1 to +1).")
    frustration_peaks: int = Field(0, ge=0, description="Number of high-negativity spikes detected.")
    question_count: int = Field(0, ge=0)
    resolution_indicators: float = Field(
        0.0, ge=0.0, le=1.0, description="Signals of resolution (thanks, 'det var bra', resolved language at end)."
    )


class CallAgentPerformance(BaseModel):
    """Per-call structured output from the agent_performance engine.

    This is what gets .model_dump()'ed and stored in CallAnalysisReport.results["agent_performance"]
    (and can be merged/compared with llm["agent_assessment"]).
    Actionable for supervisors immediately.
    """

    model_config = ConfigDict(extra="forbid")

    agent: AgentMetrics
    customer: CustomerMetrics
    # Lightweight local (rule) hints before/without LLM
    local_coaching_hints: list[str] = Field(
        default_factory=list, description="Immediate rule-based suggestions (e.g. 'Agent bör hälsa tidigare')."
    )
    evidence_summary: str | None = Field(None, description="Short Swedish summary of key metric drivers.")
