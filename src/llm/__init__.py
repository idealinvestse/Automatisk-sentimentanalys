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
from .mistral_analyzer import ConversationMistralAnalyzer
from .openrouter_client import OpenRouterClient
from .prompts import SYSTEM_PROMPT as LLM_SYSTEM_PROMPT, build_user_prompt, get_system_prompt
from .schemas import (
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
    "build_user_prompt",
    "get_system_prompt",
    "LLM_SYSTEM_PROMPT",
]
