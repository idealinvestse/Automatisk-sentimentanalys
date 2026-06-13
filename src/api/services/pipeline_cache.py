"""Per-call report caching for Fas 4 endpoints (cache-before-reanalyze)."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from ...caching import AggregateCache
from ...core.models import CallAnalysisReport
from ...pipeline import CallAnalysisPipeline

logger = logging.getLogger(__name__)


def segments_fingerprint(segments: list[dict[str, Any]]) -> str:
    canonical = json.dumps(segments, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]


def report_cache_key(pipe: CallAnalysisPipeline, segments: list[dict[str, Any]]) -> str:
    fp = segments_fingerprint(segments)
    return f"report:{pipe.profile}:{pipe.sentiment_model}:{pipe.device}:{fp}"


def resolve_reports(
    pipe: CallAnalysisPipeline,
    segments_list: list[list[dict[str, Any]]],
    *,
    reanalyze: bool = False,
) -> tuple[list[CallAnalysisReport], int]:
    """Resolve call reports, using per-call cache when ``reanalyze`` is False.

    Returns:
        Tuple of (reports, cache_hit_count).
    """
    cache: AggregateCache = pipe.cache
    reports: list[CallAnalysisReport] = []
    cache_hits = 0

    for segments in segments_list:
        key = report_cache_key(pipe, segments)
        if not reanalyze:
            cached = cache.get(key)
            if cached is not None:
                payload = {k: v for k, v in cached.items() if k not in ("computed_at", "ttl")}
                reports.append(CallAnalysisReport.from_dict(payload))
                cache_hits += 1
                continue

        report = pipe.analyze_segments(segments)
        cache.set(key, report.to_dict())
        reports.append(report)

    if cache_hits:
        logger.info(
            "resolve_reports: %d/%d call(s) served from cache (reanalyze=%s)",
            cache_hits,
            len(segments_list),
            reanalyze,
        )
    return reports, cache_hits
