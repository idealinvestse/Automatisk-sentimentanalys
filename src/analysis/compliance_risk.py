"""Compliance Risk Analyzer.

Flags potential compliance, legal or policy risks in agent language:
- Over-promising ("jag lovar", "garanterat", "absolut")
- Inappropriate data requests
- Risky statements

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


@register_analyzer("compliance_risk")
class ComplianceRiskAnalyzer(Analyzer):
    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "compliance_risk"

    @property
    def requires(self) -> list[str]:
        return ["intent"]

    def analyze(self, ctx: AnalysisContext) -> dict[str, Any]:
        if not ctx.segments:
            return {"risk_level": "low", "flagged": []}

        flagged = []
        high_risk_count = 0

        for seg in ctx.segments:
            text = (seg.text or "").lower()
            risks = []
            if any(kw in text for kw in PROMISE_KEYWORDS):
                risks.append("over_promise")
            if any(kw in text for kw in DATA_RISK):
                risks.append("data_request")

            if risks:
                high_risk_count += 1
                flagged.append({
                    "speaker": getattr(seg, "speaker", None),
                    "start": getattr(seg, "start", 0),
                    "end": getattr(seg, "end", 0),
                    "risks": risks,
                    "evidence": seg.text[:120] if seg.text else "",
                    "severity": "high" if "over_promise" in risks else "medium",
                })

        risk_level = "high" if high_risk_count > 2 else ("medium" if high_risk_count > 0 else "low")

        return {
            "overall_risk_level": risk_level,
            "flagged_segments": flagged,
            "recommendation": "Review high-risk segments for compliance training" if risk_level != "low" else "No major issues detected",
        }
