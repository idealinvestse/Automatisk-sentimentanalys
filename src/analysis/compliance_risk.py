"""Compliance Risk Analyzer.

Flags potential compliance, legal or policy risks in agent language:
- Over-promising ("jag lovar", "garanterat", "absolut")
- Inappropriate data requests
- Missing recording disclosure
- GDPR-sensitive requests
- Threatening language

Returns flagged segments + risk level + evidence.
Critical for regulated industries and QA compliance scoring.
"""

from __future__ import annotations

import logging
from typing import Any

from ..core.models import AnalysisContext
from .base import Analyzer
from .registry import register_analyzer

logger = logging.getLogger(__name__)

PROMISE_KEYWORDS = ["jag lovar", "garanterar", "absolut", "definitivt", "100%", "säker på att"]
DATA_RISK = ["skicka kort", "personnummer", "bankuppgifter", "lösenord", "konto"]
RECORDING_KEYWORDS = ["inspelat samtal", "inspelning", "bandas", "för kvalitetssäkring"]
GDPR_KEYWORDS = ["samtycke", "personuppgifter", "gdpr", "dataskydd"]
THREAT_KEYWORDS = ["polisanmäla", "advokat", "stämmer", "anmäla er", "konsumentverket"]


def _agent_speakers(role_result: dict[str, Any]) -> set[str]:
    roles = role_result.get("roles") if isinstance(role_result, dict) else {}
    if not isinstance(roles, dict):
        return set()
    return {sp for sp, role in roles.items() if str(role).lower() == "agent"}


def _empty_result() -> dict[str, Any]:
    return {
        "overall_risk_level": "low",
        "flagged_segments": [],
        "recommendation": "Inga större compliance-problem identifierade",
    }


@register_analyzer("compliance_risk")
class ComplianceRiskAnalyzer(Analyzer):
    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "compliance_risk"

    @property
    def requires(self) -> list[str]:
        return ["role"]

    def analyze(self, ctx: AnalysisContext) -> dict[str, Any]:
        if not ctx.segments:
            return _empty_result()

        role_result = ctx.results.get("role") or {}
        agent_speakers = _agent_speakers(role_result)
        flagged = []
        high_risk_count = 0

        for idx, seg in enumerate(ctx.segments):
            speaker = getattr(seg, "speaker", None)
            if agent_speakers and speaker not in agent_speakers:
                continue

            text = (seg.text or "").lower()
            risks = []
            if any(kw in text for kw in PROMISE_KEYWORDS):
                risks.append("over_promise")
            if any(kw in text for kw in DATA_RISK):
                risks.append("data_request")
            if any(kw in text for kw in RECORDING_KEYWORDS):
                risks.append("recording_disclosure")
            if any(kw in text for kw in GDPR_KEYWORDS):
                risks.append("gdpr_reference")
            if any(kw in text for kw in THREAT_KEYWORDS):
                risks.append("customer_threat")

            if risks:
                if "over_promise" in risks or "data_request" in risks:
                    high_risk_count += 1
                flagged.append({
                    "segment_index": idx,
                    "speaker": speaker,
                    "start": getattr(seg, "start", 0),
                    "end": getattr(seg, "end", 0),
                    "risks": risks,
                    "evidence": seg.text[:120] if seg.text else "",
                    "severity": "high" if "over_promise" in risks or "data_request" in risks else "medium",
                })

        risk_level = "high" if high_risk_count > 2 else ("medium" if high_risk_count > 0 else "low")
        recommendation = (
            "Granska flaggade segment för compliance-utbildning"
            if risk_level != "low"
            else "Inga större compliance-problem identifierade"
        )

        return {
            "overall_risk_level": risk_level,
            "flagged_segments": flagged,
            "recommendation": recommendation,
        }
