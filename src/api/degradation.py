"""Helpers for surfacing graceful-degradation signals to API clients."""

from __future__ import annotations

from typing import Any


def collect_degraded_reasons(report: Any) -> list[str]:
    """Derive a flat list of degraded-mode reasons from a CallAnalysisReport."""
    reasons: list[str] = []
    results = getattr(report, "results", None) or {}
    if not isinstance(results, dict):
        results = {}

    degradation = results.get("degradation")
    if isinstance(degradation, dict):
        for key in ("reasons", "skipped", "unavailable"):
            val = degradation.get(key)
            if isinstance(val, list):
                reasons.extend(str(x) for x in val if x)
            elif isinstance(val, str) and val:
                reasons.append(val)
        if degradation.get("mode") and not reasons:
            reasons.append(str(degradation["mode"]))

    llm = getattr(report, "llm", None) or {}
    if isinstance(llm, dict):
        if llm.get("llm_used") is False or llm.get("used") is False:
            note = llm.get("skip_reason") or llm.get("reason") or "llm_skipped"
            reasons.append(f"llm:{note}")
        if llm.get("fallback"):
            reasons.append("llm:fallback")

    routing = results.get("analyzer_routing")
    if isinstance(routing, dict):
        skipped = routing.get("skipped") or routing.get("unavailable") or []
        if isinstance(skipped, list):
            reasons.extend(f"analyzer_skipped:{s}" for s in skipped if s)

    partial = results.get("partial")
    if isinstance(partial, dict) and partial.get("incomplete"):
        reasons.append("partial:incomplete")

    # Deduplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out
