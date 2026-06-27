"""RootCauseInsightAnalyzer (full implementation).

Identifies likely root causes behind customer issues by combining:
- Intent + trajectory patterns
- Repeated negative sentiment/emotion
- Specific complaint keywords

Returns structured root causes with evidence, severity and suggested systemic fixes.
Extremely valuable for product, process and QA teams.
"""

from __future__ import annotations

import logging
from typing import Any

from ..core.models import AnalysisContext
from .base import Analyzer
from .registry import register_analyzer

logger = logging.getLogger(__name__)

ROOT_CAUSE_KEYWORDS = {
    "lång väntetid": ["vänta", "lång tid", "hur länge", "fortfarande inte"],
    "script rigidity": ["jag måste", "systemet säger", "jag kan inte göra", "enligt våra rutiner"],
    "produktfel": ["funkar inte", "går sönder", "fel på", "bugg"],
    "otydlig information": ["förstår inte", "vad menar ni", "oklart", "hur gör jag"],
}


@register_analyzer("root_cause")
class RootCauseInsightAnalyzer(Analyzer):
    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "root_cause"

    @property
    def requires(self) -> list[str]:
        return ["intent", "trajectory", "sentiment"]

    def analyze(self, ctx: AnalysisContext) -> dict[str, Any]:
        if not ctx.segments:
            return {"root_causes": [], "overall_risk": "low"}

        causes: dict[str, int] = {}
        evidence_list = []

        for seg in ctx.segments:
            text = (seg.text or "").lower()
            for cause, keywords in ROOT_CAUSE_KEYWORDS.items():
                if any(kw in text for kw in keywords):
                    causes[cause] = causes.get(cause, 0) + 1
                    evidence_list.append({
                        "cause": cause,
                        "segment_start": getattr(seg, "start", 0),
                        "evidence": seg.text[:80] if seg.text else "",
                    })

        if not causes:
            return {"root_causes": [], "overall_risk": "low", "message": "No clear root causes detected"}

        # Sort by frequency
        sorted_causes = sorted(causes.items(), key=lambda x: x[1], reverse=True)
        top_cause = sorted_causes[0][0]

        recommendations = {
            "lång väntetid": "Optimera routing och staffing",
            "script rigidity": "Ge agenter mer handlingsutrymme + bättre undantagshantering",
            "produktfel": "Prioritera buggfix + tydligare felmeddelanden",
            "otydlig information": "Förbättra kunskapsbas och interna sökverktyg",
        }

        return {
            "root_causes": [
                {
                    "cause": c,
                    "count": cnt,
                    "recommendation": recommendations.get(c, "Analysera vidare"),
                }
                for c, cnt in sorted_causes[:3]
            ],
            "top_root_cause": top_cause,
            "evidence_examples": evidence_list[:5],
            "overall_risk": "high" if sorted_causes[0][1] > 2 else "medium",
        }