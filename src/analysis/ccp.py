"""Critical Control Points (CCP) and living analyzer routing for the deep path."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# YAML profiles remain priors; these are runtime adjustments.
CORE_LOCAL = (
    "sentiment",
    "intent",
    "role",
    "emotion",
    "negation",
    "compliance_risk",
)
DETERMINISTIC_SENSORS = (
    "customer_effort",
    "active_listening",
    "aspect",
)


@dataclass
class CCPCheck:
    name: str
    passed: bool
    detail: str
    corrective_action: str | None = None


@dataclass
class CCPResult:
    passed: bool
    checks: list[CCPCheck] = field(default_factory=list)

    def failed_names(self) -> list[str]:
        return [c.name for c in self.checks if not c.passed]


def evaluate_deep_path_ccps(
    segments: list[Any],
    results: dict[str, Any],
    *,
    min_segments: int = 6,
    min_avg_chars: float = 12.0,
    require_sentiment: bool = True,
) -> CCPResult:
    """Named CCPs that must pass before LLM deep output is accepted."""
    checks: list[CCPCheck] = []

    # CCP-1: PII clean (redaction succeeded or not required)
    pii = results.get("pii_redaction")
    pii_ok = True
    pii_detail = "No PII redaction error"
    if isinstance(pii, dict) and pii.get("error"):
        pii_ok = False
        pii_detail = f"PII redaction failed: {pii.get('error')}"
    checks.append(
        CCPCheck(
            name="pii_clean",
            passed=pii_ok,
            detail=pii_detail,
            corrective_action="Skip LLM; fix redaction config" if not pii_ok else None,
        )
    )

    # CCP-2: Minimum segment quality
    n = len(segments or [])
    texts = []
    for seg in segments or []:
        if isinstance(seg, dict):
            texts.append(str(seg.get("text") or ""))
        else:
            texts.append(str(getattr(seg, "text", "") or ""))
    avg_chars = (sum(len(t.strip()) for t in texts) / n) if n else 0.0
    quality_ok = n >= min_segments and avg_chars >= min_avg_chars
    checks.append(
        CCPCheck(
            name="min_segment_quality",
            passed=quality_ok,
            detail=f"segments={n} avg_chars={avg_chars:.1f} (min_segments={min_segments}, min_avg={min_avg_chars})",
            corrective_action="Skip LLM; wait for more/better ASR" if not quality_ok else None,
        )
    )

    # CCP-3: Sentiment / negation sanity — sentiment must exist when deep path runs
    sent = results.get("sentiment")
    neg = results.get("negation")
    sanity_ok = True
    sanity_detail = "Sentiment/negation present"
    if require_sentiment:
        if not isinstance(sent, list) or len(sent) == 0:
            sanity_ok = False
            sanity_detail = "Sentiment missing or empty — core local path required before LLM"
        elif isinstance(sent, list) and sent:
            labels = [s.get("label") if isinstance(s, dict) else None for s in sent]
            if all(l is None for l in labels):
                sanity_ok = False
                sanity_detail = "Sentiment labels all missing"
    if sanity_ok and isinstance(sent, list) and sent and isinstance(neg, list):
        if len(neg) != len(sent):
            sanity_detail = f"Negation length mismatch ({len(neg)} vs {len(sent)}) — warning only"
    checks.append(
        CCPCheck(
            name="sentiment_negation_sanity",
            passed=sanity_ok,
            detail=sanity_detail,
            corrective_action="Skip LLM; re-run core local analyzers" if not sanity_ok else None,
        )
    )

    passed = all(c.passed for c in checks)
    if not passed:
        logger.warning("Deep-path CCP failed: %s", [c.name for c in checks if not c.passed])
    return CCPResult(passed=passed, checks=checks)


def ccp_result_to_dict(ccp: CCPResult) -> dict[str, Any]:
    return {
        "passed": ccp.passed,
        "checks": [
            {
                "name": c.name,
                "passed": c.passed,
                "detail": c.detail,
                "corrective_action": c.corrective_action,
            }
            for c in ccp.checks
        ],
        "failed": ccp.failed_names(),
    }


def select_analyzers_runtime(
    profile_selected: list[str] | None,
    *,
    segment_count: int,
    intent_results: Any = None,
    compliance_risk: Any = None,
    predictive: Any = None,
) -> list[str]:
    """Living routing: adjust analyzer set from call features; YAML list is the prior."""
    base = list(profile_selected or list(CORE_LOCAL) + list(DETERMINISTIC_SENSORS))
    selected = list(dict.fromkeys(base))  # stable unique

    # Short calls: keep core + aspect only
    if segment_count < 4:
        keep = set(CORE_LOCAL) | {"aspect", "customer_effort"}
        selected = [a for a in selected if a in keep]
        return selected

    # Intent mix: complaint-heavy → ensure compliance + resolution signals
    intents: list[str] = []
    if isinstance(intent_results, list):
        for item in intent_results:
            if isinstance(item, dict):
                intents.append(str(item.get("intent", "")).lower())
            elif isinstance(item, (list, tuple)) and item:
                intents.append(str(item[0]).lower())
    complaint_like = sum(1 for i in intents if any(k in i for k in ("klagomål", "complaint", "reklamation", "fel")))
    if complaint_like >= max(1, len(intents) // 3):
        for extra in ("compliance_risk", "resolution_probability", "predictive"):
            if extra not in selected:
                selected.append(extra)

    # Elevated risk → add predictive / multi_turn if missing
    risk_level = None
    if isinstance(compliance_risk, dict):
        risk_level = compliance_risk.get("overall_risk_level")
    if isinstance(predictive, dict) and predictive.get("risk_level") in ("high", "critical"):
        risk_level = "high"
    if risk_level in ("medium", "high", "critical"):
        for extra in ("predictive", "multi_turn_journey"):
            if extra not in selected:
                selected.append(extra)

    # Long calls: enable summary/topics
    if segment_count >= 10:
        for extra in ("summary", "topics"):
            if extra not in selected:
                selected.append(extra)

    return selected
