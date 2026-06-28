"""Explicit pipeline steps extracted from CallAnalysisPipeline (DEBT-10 / PIPE-01)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .analysis import resolve_analyzers_for_profile, run_analyzers, run_analyzers_async
from .core.models import AnalysisContext, Segment

logger = logging.getLogger(__name__)

__all__ = [
    "PipelineLLMContext",
    "apply_early_pii_redaction",
    "run_registry_analyzers",
    "run_registry_analyzers_async",
    "should_use_any_llm",
    "run_llm_holistic",
    "run_fas4_enrichment",
]


@dataclass
class PipelineLLMContext:
    """LLM routing configuration shared by Fas-4 enrichment steps."""

    profile: str
    provider: str
    use_mistral_llm: bool
    deep_analysis: bool
    llm_model: str | None
    llm_api_key: str | None
    groq_eu_residency: bool


def apply_early_pii_redaction(
    segments: list[Segment],
    *,
    profile_name: str,
) -> tuple[list[Segment], Any | None]:
    """Fas 4.4.1 early PII redaction before analyzers/LLM."""
    try:
        from .llm.pii_redactor import redact_segments

        seg_dicts = [s.to_dict() for s in segments]
        redacted_dicts, pii_log = redact_segments(seg_dicts, profile_name=profile_name, return_log=True)
        if pii_log and pii_log.total_redacted > 0:
            logger.info(
                "PII redaction (early): %d events, types=%s",
                pii_log.total_redacted,
                pii_log.types_redacted,
            )
            segments = [Segment.from_dict(d) for d in redacted_dicts]
        return segments, pii_log
    except Exception as exc:
        logger.debug("Early PII redaction skipped: %s", exc)
        return segments, None


async def run_registry_analyzers_async(
    segments: list[Segment],
    *,
    profile: str,
    selected_analyzers: list[str] | None,
    analyzer_configs: dict[str, dict[str, Any]],
    transcript: Any | None = None,
) -> dict[str, Any]:
    """Async registry execution for FastAPI routes."""
    ctx = AnalysisContext(transcript=transcript, segments=segments)
    resolved = resolve_analyzers_for_profile(profile, explicit_selected=selected_analyzers)
    return await run_analyzers_async(ctx, selected=resolved, analyzer_configs=analyzer_configs)


def run_registry_analyzers(
    segments: list[Segment],
    *,
    profile: str,
    selected_analyzers: list[str] | None,
    analyzer_configs: dict[str, dict[str, Any]],
    async_mode: bool = False,
    transcript: Any | None = None,
) -> dict[str, Any]:
    """Run the analyzer registry on segments (sync entry point)."""
    ctx = AnalysisContext(transcript=transcript, segments=segments)
    resolved = resolve_analyzers_for_profile(profile, explicit_selected=selected_analyzers)
    return run_analyzers(
        ctx,
        selected=resolved,
        analyzer_configs=analyzer_configs,
        async_mode=async_mode,
    )


def _segments_to_dicts(segments: list[Segment] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    seg_dicts: list[dict[str, Any]] = []
    for segment in segments:
        if isinstance(segment, dict):
            seg_dicts.append(segment)
        else:
            seg_dicts.append(segment.to_dict())
    return seg_dicts


def should_use_mistral_llm(segments: list, ctx: PipelineLLMContext) -> bool:
    """Decision logic for hybrid path (profile + length + confidence + explicit flags)."""
    if ctx.deep_analysis or ctx.use_mistral_llm:
        return True
    return ctx.profile in {"callcenter", "call", "customer_service"} and len(segments) >= 6


def should_use_groq_llm(segments: list, ctx: PipelineLLMContext) -> bool:
    """Decision logic for Groq hybrid path."""
    if ctx.provider != "groq":
        return False
    if ctx.deep_analysis or ctx.use_mistral_llm:
        return True
    return ctx.profile in {"callcenter", "call", "customer_service"} and len(segments) >= 6


def should_use_any_llm(segments: list, ctx: PipelineLLMContext) -> bool:
    """Unified decision: should we use ANY LLM path based on provider?"""
    if ctx.provider == "groq":
        return should_use_groq_llm(segments, ctx)
    return should_use_mistral_llm(segments, ctx)


def run_mistral_holistic(
    segments: list[Segment] | list[dict[str, Any]],
    results: dict[str, Any],
    ctx: PipelineLLMContext,
) -> dict[str, Any]:
    """Call the Mistral analyzer (if available) and return enriched result or fallback dict."""
    try:
        from .llm.mistral_analyzer import ConversationMistralAnalyzer

        role_map = results.get("role") or {}
        seg_dicts = _segments_to_dicts(segments)
        mistral = ConversationMistralAnalyzer(
            model=ctx.llm_model,
            api_key=ctx.llm_api_key,
        )
        llm_out = mistral.analyze_full_conversation(
            segments=seg_dicts,
            role_map=role_map if isinstance(role_map, dict) else {},
            local_results=results,
            profile_name=ctx.profile,
        )
        if llm_out.get("fallback"):
            llm_out["meta"] = llm_out.get("meta", {})
            llm_out["meta"]["llm_used"] = False
            llm_out["meta"]["llm_fallback_reason"] = llm_out.get("meta", {}).get(
                "fallback_reason", "llm_error_or_disabled"
            )
        else:
            llm_out.setdefault("meta", {})
            llm_out["meta"]["llm_used"] = True
        return llm_out
    except Exception as exc:
        logger.warning("Mistral holistic step failed (will use local only): %s", exc)
        return {"llm_used": False, "llm_fallback_reason": str(exc), "error": str(exc)}


def run_groq_holistic(
    segments: list[Segment] | list[dict[str, Any]],
    results: dict[str, Any],
    ctx: PipelineLLMContext,
) -> dict[str, Any]:
    """Call the Groq analyzer (if available) and return enriched result or fallback dict.

    GDPR gate: groq_eu_residency must be True or anonymize_before_llm must be active.
    """
    try:
        from .llm.groq_analyzer import GroqAnalyzer

        role_map = results.get("role") or {}
        seg_dicts = _segments_to_dicts(segments)
        pii_redacted = bool(
            results.get("pii_redaction", {}).get("total_redacted", 0) > 0
            if isinstance(results.get("pii_redaction"), dict)
            else False
        )

        if not ctx.groq_eu_residency and not pii_redacted:
            logger.warning(
                "GROQ GDPR GATE: groq_eu_residency=OFF and no PII redaction detected. "
                "Groq data centers are US/Saudi Arabia (no EU hosting). "
                "Falling back to local analysis only."
            )
            return {
                "llm_used": False,
                "meta": {
                    "llm_used": False,
                    "llm_fallback_reason": "groq_gdpr_gate",
                    "provider": "groq",
                },
            }

        groq_analyzer = GroqAnalyzer(
            model=ctx.llm_model,
            api_key=ctx.llm_api_key,
            groq_eu_residency=ctx.groq_eu_residency,
        )
        llm_out = groq_analyzer.analyze_full_conversation(
            segments=seg_dicts,
            role_map=role_map if isinstance(role_map, dict) else {},
            local_results=results,
            profile_name=ctx.profile,
            anonymize_before_llm=pii_redacted,
        )
        if llm_out.get("fallback"):
            llm_out["meta"] = llm_out.get("meta", {})
            llm_out["meta"]["llm_used"] = False
            llm_out["meta"]["llm_fallback_reason"] = llm_out.get("meta", {}).get(
                "fallback_reason", "groq_llm_error_or_disabled"
            )
        else:
            llm_out.setdefault("meta", {})
            llm_out["meta"]["llm_used"] = True
            llm_out["meta"]["provider"] = "groq"
        return llm_out
    except Exception as exc:
        logger.warning("Groq holistic step failed (will use local only): %s", exc)
        return {
            "llm_used": False,
            "llm_fallback_reason": str(exc),
            "error": str(exc),
            "meta": {"provider": "groq"},
        }


def run_llm_holistic(
    segments: list[Segment] | list[dict[str, Any]],
    results: dict[str, Any],
    ctx: PipelineLLMContext,
) -> dict[str, Any]:
    """Route to the correct LLM analyzer based on provider."""
    if ctx.provider == "groq":
        return run_groq_holistic(segments, results, ctx)
    return run_mistral_holistic(segments, results, ctx)


def run_fas4_enrichment(
    segments: list[Segment],
    results: dict[str, Any],
    ctx: PipelineLLMContext,
) -> dict[str, Any]:
    """Run Fas 4 enrichment steps shared by audio and segment analysis."""
    role_res = results.get("role") or {}
    role_map = role_res.get("roles", role_res) if isinstance(role_res, dict) else {}

    try:
        from .agent_performance import compute_call_agent_performance

        sent_res = results.get("sentiment") or []
        agent_perf = compute_call_agent_performance(
            segments=segments or [],
            role_map=role_map if isinstance(role_map, dict) else {},
            sentiment_results=sent_res,
            profile_name=ctx.profile,
        )
        results["agent_performance"] = agent_perf.model_dump()
        local_assess = {
            "empathy_score": agent_perf.agent.empathy_score,
            "compliance_flags": agent_perf.agent.compliance_flags,
            "strengths": [],
            "weaknesses": [],
            "specific_coaching_recommendations": [],
            "overall_assessment": None,
            "source": "local_rules_fas4.1",
            "talk_listen_ratio": agent_perf.agent.talk_listen_ratio,
            "intervention_count": agent_perf.agent.intervention_count,
            "evidence_spans": [],
        }
        results["agent_assessment_local"] = local_assess
        results["agent_assessment"] = local_assess
        results["customer_metrics"] = agent_perf.customer.model_dump()
        logger.info(
            "Fas 4.1 agent_performance computed | empathy=%.2f flags=%s talk_ratio=%.2f",
            agent_perf.agent.empathy_score,
            agent_perf.agent.compliance_flags,
            agent_perf.agent.talk_ratio,
        )
    except Exception as exc:
        logger.warning("Fas 4.1 agent_performance step failed (non-fatal): %s", exc)
        results["agent_performance"] = {"error": str(exc), "fallback": True}

    llm_result: dict[str, Any] = {}
    if should_use_any_llm(segments or [], ctx):
        llm_result = run_llm_holistic(segments or [], results, ctx)
        results["llm"] = llm_result
        if llm_result.get("meta", {}).get("llm_used"):
            logger.info(
                "Pipeline used %s LLM for holistic analysis (model=%s, cached=%s)",
                ctx.provider,
                llm_result.get("meta", {}).get("model"),
                llm_result.get("meta", {}).get("cached"),
            )

    llm_assess = (results.get("llm") or {}).get("agent_assessment")
    if llm_assess and isinstance(llm_assess, dict) and llm_assess.get("empathy_score") is not None:
        results["agent_assessment"] = llm_assess
        logger.debug("Merged LLM agent_assessment into results (hybrid coaching available)")

    try:
        from .compliance_qa import score_call_with_default_scorecard

        use_llm_qa = bool(
            ctx.use_mistral_llm or (results.get("llm") or {}).get("meta", {}).get("llm_used")
        )
        qa_analyzer = None
        if use_llm_qa:
            if ctx.provider == "groq":
                from .llm.groq_analyzer import GroqAnalyzer

                qa_analyzer = GroqAnalyzer(
                    model=ctx.llm_model,
                    api_key=ctx.llm_api_key,
                    groq_eu_residency=ctx.groq_eu_residency,
                )
            else:
                from .llm.mistral_analyzer import ConversationMistralAnalyzer

                qa_analyzer = ConversationMistralAnalyzer(
                    model=ctx.llm_model,
                    api_key=ctx.llm_api_key,
                )
        qa_res = score_call_with_default_scorecard(
            segments=segments or [],
            role_map=role_map if isinstance(role_map, dict) else {},
            local_signals={
                "agent_performance": results.get("agent_performance"),
                "agent_assessment": results.get("agent_assessment"),
            },
            profile_name=ctx.profile,
            use_llm=use_llm_qa,
            analyzer=qa_analyzer,
        )
        results["qa"] = qa_res
        results["compliance_qa"] = qa_res
        if isinstance(qa_res, dict) and qa_res.get("llm_criteria_used"):
            logger.info(
                "Fas 4.2 QA scoring used LLM for criteria=%s (hybrid)",
                qa_res.get("llm_criteria_used"),
            )
        logger.info(
            "Fas 4.2 QA complete | score=%.1f passed=%s risk=%s",
            qa_res.get("overall_qa_score", 0) if isinstance(qa_res, dict) else 0,
            qa_res.get("passed") if isinstance(qa_res, dict) else False,
            qa_res.get("risk_level") if isinstance(qa_res, dict) else "?",
        )
    except Exception as exc:
        logger.warning("Fas 4.2 QA scoring failed (non-fatal): %s", exc)
        results["qa"] = {"error": str(exc), "overall_qa_score": 0.0, "passed": False}

    try:
        from .alerting import run_alerts_on_results

        alerts = run_alerts_on_results(results)
        if alerts:
            results["alerts"] = alerts
            logger.info("Fas 4.4.2 alerts triggered: %d (highest severity in first)", len(alerts))
    except Exception as exc:
        logger.warning("Fas 4.4.2 alerting failed (non-fatal): %s", exc)

    return llm_result
