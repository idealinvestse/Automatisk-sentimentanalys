"""LLM integration layer (Mistral/OpenRouter) for holistic post-transcription analysis.

This package provides the European-first (Mistral via OpenRouter) deep analysis path
for call center conversations. It is used *selectively* on top of the local fast-path
analyzers (see src/analysis/*) to deliver:

- Full-conversation trajectory and escalation reasoning
- Root cause analysis
- Actionable QA recommendations
- Agent performance assessment with evidence spans

All external calls are logged for GDPR traceability. Local models + heuristics remain
the default. Strict JSON schema enforcement + caching + fallback ensure reliability
and cost control.

Exposed primarily via:
- OpenRouterClient: low-level cached client
- ConversationMistralAnalyzer (in mistral_analyzer): high-level orchestrator (Fas 3.1.2+)

See UTVECKLINGSPLAN_Mistral_OpenRouter_LLM_Integration.md for roadmap and privacy notes.
"""

from __future__ import annotations

# Re-exports (populated as modules are implemented)
from .groq_analyzer import GroqAnalyzer
from .groq_client import GroqClient, get_groq_api_key
from .mistral_analyzer import ConversationMistralAnalyzer
from .openrouter_client import OpenRouterClient, get_openrouter_api_key, load_openrouter_key_from_file
from .pii_redactor import redact_pii, redact_segments
from .prompts import SYSTEM_PROMPT as LLM_SYSTEM_PROMPT, build_user_prompt, get_system_prompt
from .schemas import (
    GROQ_DEFAULT_MODEL,
    GROQ_FALLBACK_CHAIN,
    GROQ_MODELS,
    GROQ_PROD_MODELS,
    ActionableSummary,
    AgentAssessment,
    AspectItem,
    CallLLMOutput,
    EmotionTrajectoryPoint,
    EvidenceSpan,
    LLM_OUTPUT_JSON_SCHEMA,
    RootCause,
    Trajectory,
)

__all__ = [
    "OpenRouterClient",
    "get_openrouter_api_key",
    "load_openrouter_key_from_file",
    "GroqClient",
    "get_groq_api_key",
    "GroqAnalyzer",
    "ConversationMistralAnalyzer",
    "CallLLMOutput",
    "Trajectory",
    "AspectItem",
    "RootCause",
    "ActionableSummary",
    "AgentAssessment",
    "EmotionTrajectoryPoint",
    "EvidenceSpan",
    "LLM_OUTPUT_JSON_SCHEMA",
    "GROQ_MODELS",
    "GROQ_DEFAULT_MODEL",
    "GROQ_FALLBACK_CHAIN",
    "GROQ_PROD_MODELS",
    "build_user_prompt",
    "get_system_prompt",
    "LLM_SYSTEM_PROMPT",
    "redact_pii",
    "redact_segments",
]
