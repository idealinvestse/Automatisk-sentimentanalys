"""Explicit pipeline steps extracted from CallAnalysisPipeline (DEBT-10)."""

from __future__ import annotations

import logging
from typing import Any

from .core.models import AnalysisContext, Segment
from .analysis import resolve_analyzers_for_profile, run_analyzers, run_analyzers_async

logger = logging.getLogger(__name__)


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