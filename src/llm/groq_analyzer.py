"""GroqAnalyzer – high-level Groq-powered holistic call analysis.

Mirrors ConversationMistralAnalyzer but uses GroqClient instead of OpenRouterClient.

Why Groq here (and why this class):
- Groq offers extremely fast inference (~840 tps for Llama 8B) at competitive pricing.
- This is an alternative to Mistral/OpenRouter for users who want maximum speed.
- ⚠️ GDPR WARNING: Groq data centers are US + Saudi Arabia — NO confirmed EU hosting.
  Early PII redaction is MANDATORY before any Groq call.
- The class is deliberately structured identically to ConversationMistralAnalyzer
  to keep the pipeline integration minimal (same interface, different client).

Callcenter goal connection:
Same as Mistral — provides trajectory, root cause, actionable QA recommendations,
and agent assessment. Compatible as a drop-in alternative.

Usage:
    from src.llm.groq_analyzer import GroqAnalyzer
    from src.llm.groq_client import GroqClient

    analyzer = GroqAnalyzer()
    result = analyzer.analyze_full_conversation(
        segments=[{"speaker": "SPEAKER_0", "text": "..."}, ...],
        role_map={"SPEAKER_0": "agent", "SPEAKER_1": "customer"},
        tasks=["trajectory", "actionable_summary"],
    )
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from ..core.models import Segment
from .groq_client import GroqClient
from .pii_redactor import redact_segments
from .prompts import build_user_prompt, get_system_prompt
from .schemas import CallLLMOutput, GROQ_DEFAULT_MODEL, LLM_OUTPUT_JSON_SCHEMA

logger = logging.getLogger(__name__)


def _build_role_labeled_transcript(
    segments: list[dict[str, Any]] | list[Segment],
    role_map: dict[str, str] | None = None,
) -> str:
    """Turn segments into a clean, role-aware transcript for the LLM."""
    lines: list[str] = []
    for seg in segments:
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


class GroqAnalyzer:
    """Orchestrates full-conversation Groq analysis with strict JSON output.

    See module docstring for rationale, GDPR warning, and callcenter value.
    """

    SUPPORTED_TASKS = [
        "trajectory",
        "refined_aspects",
        "root_cause",
        "actionable_summary",
        "agent_assessment",
        "agent_assessment_detailed",
        "emotion_trajectory",
    ]

    def __init__(
        self,
        client: GroqClient | None = None,
        model: str | None = None,
        temperature: float = 0.15,
        max_tokens: int = 4096,
        api_key: str | None = None,
        # GDPR gate
        groq_eu_residency: bool = False,
    ) -> None:
        if client is not None:
            self.client = client
        else:
            self.client = GroqClient(
                api_key=api_key,
                default_model=model or GROQ_DEFAULT_MODEL,
                groq_eu_residency=groq_eu_residency,
            )
        self.model = model or self.client.default_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.groq_eu_residency = groq_eu_residency

    def analyze_full_conversation(
        self,
        segments: list[dict[str, Any]] | list[Segment],
        role_map: dict[str, str] | None = None,
        tasks: list[str] | None = None,
        local_results: dict[str, Any] | None = None,
        profile_name: str = "callcenter",
        # Derive anonymize_before_llm from profile; caller can override
        anonymize_before_llm: bool | None = None,
    ) -> dict[str, Any]:
        """Run holistic analysis on the full conversation via Groq.

        Args:
            segments: List of segment dicts or Segment objects.
            role_map: Optional mapping SPEAKER_X → "agent" | "customer".
            tasks: Subset of SUPPORTED_TASKS. If None, runs all.
            local_results: Optional dict of already computed local analysis.
            profile_name: For logging / cost attribution.
            anonymize_before_llm: Whether PII was redacted before this call.
                Auto-derived from profile if None.

        Returns:
            Dict validated from CallLLMOutput + top-level "meta".
            On failure: minimal fallback dict with "error" and "fallback": True.
        """
        if not segments:
            return {"error": "no_segments", "fallback": True}

        tasks = tasks or self.SUPPORTED_TASKS
        tasks = [t for t in tasks if t in self.SUPPORTED_TASKS]

        # Determine anonymize_before_llm from profile if not explicitly set
        if anonymize_before_llm is None:
            anonymize_before_llm = _check_profile_anonymize(profile_name)

        # PII redaction before sending to Groq (GDPR: US/Saudi data centers)
        segments_for_llm = redact_segments(segments, profile_name=profile_name)
        # If redact_segments didn't "change" the list, we still might have unredacted PII;
        # the anonymize_before_llm flag controls whether we consider data safe.
        # For Groq (no EU residency), always prefer redaction.
        anonymized = anonymize_before_llm or (segments_for_llm is not segments)

        transcript = _build_role_labeled_transcript(segments_for_llm, role_map)
        transcript_hash = _make_transcript_hash(transcript, role_map)

        # Lightweight local context (same as Mistral)
        local_ctx = {}
        if local_results:
            local_ctx = {
                "role_inference": (local_results.get("role") or {}).get("roles")
                if isinstance(local_results.get("role"), dict)
                else local_results.get("role"),
                "sentiment_summary": self._summarize_sentiment(local_results.get("sentiment")),
                "escalation_from_local": local_results.get("trajectory", {}).get("escalation_events"),
                "agent_performance_local": local_results.get("agent_performance")
                or local_results.get("agent_assessment_local"),
            }

        local_ctx_str = (
            json.dumps(local_ctx, ensure_ascii=False, indent=2) if local_ctx else "Ingen tidigare analys."
        )
        user_prompt = build_user_prompt(
            transcript, local_context={"summary": local_ctx_str}, tasks=tasks
        )

        messages = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": user_prompt},
        ]

        try:
            schema = LLM_OUTPUT_JSON_SCHEMA

            result_dict, meta = self.client.structured_chat(
                messages=messages,
                json_schema=schema,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                task_name="full_holistic_call_analysis",
                transcript_hash=transcript_hash,
                anonymize_before_llm=anonymized,
            )

            # Validate + normalize
            validated = CallLLMOutput.model_validate(result_dict)

            # Merge client meta
            output = validated.model_dump()
            output["meta"] = {
                **validated.meta,
                **meta,
                "profile": profile_name,
                "tasks": tasks,
                "pii_redacted": bool(anonymized),
                "provider": "groq",
            }

            if "model" not in output["meta"]:
                output["meta"]["model"] = meta.get("model", self.model)

            logger.info(
                "Groq holistic analysis complete | model=%s | cached=%s | cost≈$%s | tasks=%s",
                output["meta"].get("model"),
                meta.get("cached"),
                meta.get("cost_usd"),
                tasks,
            )
            return output

        except Exception as e:
            logger.error("Groq analyzer failed, returning fallback: %s", e, exc_info=True)
            return {
                "error": str(e),
                "fallback": True,
                "meta": {
                    "model": self.model,
                    "fallback_reason": "groq_llm_error",
                    "profile": profile_name,
                    "provider": "groq",
                },
            }

    @staticmethod
    def _summarize_sentiment(sentiment_results: Any) -> dict[str, Any]:
        if not isinstance(sentiment_results, list):
            return {}
        neg = sum(
            1
            for s in sentiment_results
            if isinstance(s, dict) and s.get("label") in ("negativ", "negative")
        )
        return {
            "count": len(sentiment_results),
            "negative_count": neg,
            "negative_ratio": round(neg / max(1, len(sentiment_results)), 3),
        }


def _check_profile_anonymize(profile_name: str) -> bool:
    """Check if profile has anonymize_before_llm enabled."""
    try:
        from ..profiles import resolve_profile
        _, spec = resolve_profile(profile=profile_name)
        llm_spec = (spec or {}).get("llm", {}) or {}
        return bool(llm_spec.get("anonymize_before_llm", False))
    except Exception:
        return False