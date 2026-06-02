"""ConversationMistralAnalyzer – high-level Mistral-powered holistic analyzer (Task 3.1.2).

This is the main entry point for the "deep path" in the hybrid architecture.

Why Mistral via OpenRouter here (and why this class):
- Local models (XLM-R sentiment, heuristics, small trajectory) are fast, cheap, offline and
  excellent for per-segment work. They lack reliable long-range reasoning, implicit meaning,
  sarcasm detection across turns, and high-quality "actionable for QA" synthesis on real
  Swedish callcenter data.
- Mistral Medium 3.5 (and Large 3) are currently among the strongest European models with
  excellent Swedish performance and long context (256k). Routing them through OpenRouter
  gives us a single API, good uptime, and easier future switch to self-hosted Mistral
  without changing code.
- The class is deliberately *not* an `analysis.base.Analyzer` yet. That integration
  (dependency on role/sentiment/emotion, registration, ctx.results["llm"]) happens in
  Task 3.2.2. This keeps Fas 3.1 focused on getting reliable LLM output quickly.
- Strict Pydantic validation after every LLM response guarantees that whatever we merge
  later into CallAnalysisReport is well-typed and auditable.
- Caching, cost meta, and the mandatory external-call log all live in the client; this
  class only orchestrates prompt construction and result shaping.

Callcenter goal connection:
The output is designed so that a QA manager or team lead can open a call report and
immediately see:
- How the customer's emotion evolved and where it broke
- What the *real* problem was (root cause, not the first complaint)
- Concrete, coachable recommendations for the specific agent on this call
- Evidence the LLM can be challenged on (spans + speaker roles)

European / GDPR:
- Never instantiated or called unless the active profile has llm.enabled or the user
  passed --use-mistral-llm / deep_analysis.
- Every execution path that reaches the client emits the "data sent externally" log.
- The schema and meta make it trivial for downstream consumers (dashboard, export) to
  tag results as "LLM-enhanced (OpenRouter/Mistral – external processing)".
- Later Fas 3.4 will add optional PII redaction before the transcript is even built.

Usage (after client is configured):
    from src.llm.mistral_analyzer import ConversationMistralAnalyzer
    from src.llm import OpenRouterClient

    analyzer = ConversationMistralAnalyzer()
    result = analyzer.analyze_full_conversation(
        segments=[{"speaker": "SPEAKER_0", "text": "..."}, ...],
        role_map={"SPEAKER_0": "agent", "SPEAKER_1": "customer"},
        tasks=["trajectory", "actionable_summary", "agent_assessment", "agent_assessment_detailed"],
    )
    # result is a dict (from CallLLMOutput) + "meta" with cost, model, cached etc.

The prompts used in this phase are solid but will be moved to src/llm/prompts.py and
further tuned in Task 3.2.1 (iterative testing on Swedish callcenter samples).
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from ..core.models import Segment
from .openrouter_client import OpenRouterClient
from .pii_redactor import redact_segments
from .prompts import build_user_prompt, get_system_prompt
from .schemas import CallLLMOutput, LLM_OUTPUT_JSON_SCHEMA

logger = logging.getLogger(__name__)


def _build_role_labeled_transcript(
    segments: list[dict[str, Any]] | list[Segment],
    role_map: dict[str, str] | None = None,
) -> str:
    """Turn segments into a clean, role-aware transcript for the LLM."""
    lines: list[str] = []
    for i, seg in enumerate(segments):
        if isinstance(seg, dict):
            text = seg.get("text", "").strip()
            speaker = seg.get("speaker") or seg.get("speaker_label") or "UNKNOWN"
        else:
            text = getattr(seg, "text", "").strip()
            speaker = getattr(seg, "speaker", None) or "UNKNOWN"

        role = "UNKNOWN"
        if role_map and speaker in role_map:
            role = role_map[speaker].upper()
        elif speaker and "agent" in str(speaker).lower():
            role = "AGENT"
        elif speaker and "customer" in str(speaker).lower():
            role = "CUSTOMER"

        prefix = f"[{role}]" if role != "UNKNOWN" else f"[{speaker}]"
        lines.append(f"{prefix} {text}")

    return "\n".join(lines)


def _make_transcript_hash(transcript: str, role_map: dict | None) -> str:
    payload = transcript + json.dumps(role_map or {}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class ConversationMistralAnalyzer:
    """Orchestrates full-conversation Mistral analysis via OpenRouter with strict schemas.

    See module docstring for extensive rationale, callcenter value and privacy considerations.
    """

    SUPPORTED_TASKS = [
        "trajectory",
        "refined_aspects",
        "root_cause",
        "actionable_summary",
        "agent_assessment",
        "agent_assessment_detailed",  # Fas 4.1.2: richer coaching recs + evidence (uses same AgentAssessment schema)
        "emotion_trajectory",
    ]

    def __init__(
        self,
        client: OpenRouterClient | None = None,
        model: str | None = None,
        temperature: float = 0.15,
        max_tokens: int = 4096,
        api_key: str | None = None,   # convenience: pass key directly (e.g. from dashboard override)
    ) -> None:
        if client is not None:
            self.client = client
        else:
            self.client = OpenRouterClient(api_key=api_key) if api_key else OpenRouterClient()
        self.model = model or self.client.default_model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def analyze_full_conversation(
        self,
        segments: list[dict[str, Any]] | list[Segment],
        role_map: dict[str, str] | None = None,
        tasks: list[str] | None = None,
        local_results: dict[str, Any] | None = None,
        profile_name: str = "callcenter",
    ) -> dict[str, Any]:
        """Run holistic analysis on the full conversation.

        Args:
            segments: List of segment dicts or Segment objects (must contain at least "text").
            role_map: Optional mapping SPEAKER_X -> "agent" | "customer".
            tasks: Subset of SUPPORTED_TASKS. If None, runs all.
            local_results: Optional dict of already computed local analysis (sentiment etc.)
                           to give the LLM as context (helps consistency).
            profile_name: For future logging / cost attribution.

        Returns:
            Dict that is a validated CallLLMOutput + top-level "meta" (cost, cached, model...).
            On any hard failure: returns a minimal fallback dict with "error" and "fallback": true
            so callers can always continue with local-only data.
        """
        if not segments:
            return {"error": "no_segments", "fallback": True}

        tasks = tasks or self.SUPPORTED_TASKS
        tasks = [t for t in tasks if t in self.SUPPORTED_TASKS]

        # Optional PII redaction before any external LLM call (Fas 3.4 follow-up)
        segments_for_llm = redact_segments(segments, profile_name=profile_name)
        anonymized = segments_for_llm is not segments  # simple heuristic: if list identity changed or content redacted

        transcript = _build_role_labeled_transcript(segments_for_llm, role_map)
        transcript_hash = _make_transcript_hash(transcript, role_map)

        # Lightweight local context (do not send huge objects)
        local_ctx = {}
        if local_results:
            # Only forward small, high-value signals
            local_ctx = {
                "role_inference": (local_results.get("role") or {}).get("roles") if isinstance(local_results.get("role"), dict) else local_results.get("role"),
                "sentiment_summary": self._summarize_sentiment(local_results.get("sentiment")),
                "escalation_from_local": local_results.get("trajectory", {}).get("escalation_events"),
                # Fas 4.1.1 + 4.1.2: forward local quantitative metrics so LLM can produce evidence-based coaching merged with numbers
                "agent_performance_local": local_results.get("agent_performance") or local_results.get("agent_assessment_local"),
            }

        local_ctx_str = json.dumps(local_ctx, ensure_ascii=False, indent=2) if local_ctx else "Ingen tidigare analys."
        user_prompt = build_user_prompt(transcript, local_context={"summary": local_ctx_str}, tasks=tasks)

        messages = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": user_prompt},
        ]

        try:
            # Use the full CallLLMOutput schema for strong guidance + validation
            schema = LLM_OUTPUT_JSON_SCHEMA

            result_dict, meta = self.client.structured_chat(
                messages=messages,
                json_schema=schema,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                task_name="full_holistic_call_analysis",
                transcript_hash=transcript_hash,
            )

            # Validate + normalize
            validated = CallLLMOutput.model_validate(result_dict)

            # Merge client meta into the model meta
            output = validated.model_dump()
            output["meta"] = {**validated.meta, **meta, "profile": profile_name, "tasks": tasks, "pii_redacted": bool(anonymized)}

            # Ensure we always have the top-level model key for convenience
            if "model" not in output["meta"]:
                output["meta"]["model"] = meta.get("model", self.model)

            logger.info(
                "Mistral holistic analysis complete | model=%s | cached=%s | cost≈$%s | tasks=%s",
                output["meta"].get("model"),
                meta.get("cached"),
                meta.get("cost_usd"),
                tasks,
            )
            return output

        except Exception as e:
            logger.error("Mistral analyzer failed, returning fallback: %s", e, exc_info=True)
            return {
                "error": str(e),
                "fallback": True,
                "meta": {
                    "model": self.model,
                    "fallback_reason": "llm_error",
                    "profile": profile_name,
                },
            }

    @staticmethod
    def _summarize_sentiment(sentiment_results: Any) -> dict[str, Any]:
        if not isinstance(sentiment_results, list):
            return {}
        neg = sum(1 for s in sentiment_results if isinstance(s, dict) and s.get("label") in ("negativ", "negative"))
        return {
            "count": len(sentiment_results),
            "negative_count": neg,
            "negative_ratio": round(neg / max(1, len(sentiment_results)), 3),
        }
