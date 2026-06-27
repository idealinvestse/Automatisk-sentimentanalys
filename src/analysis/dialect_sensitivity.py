"""DialectSensitivityAnalyzer.

Detects regional Swedish dialects, slang and ASR-challenging language (especially relevant for Dalarna, Norrland, etc.).
Flags segments with potential low ASR accuracy due to dialect and suggests better handling.

Also detects common Swedish informal/slang expressions that may affect sentiment/intent accuracy.
"""

from __future__ import annotations

import logging
from typing import Any

from ..core.models import AnalysisContext
from .base import Analyzer
from .registry import register_analyzer

logger = logging.getLogger(__name__)

DIALECT_MARKERS = {
    "dalarna/norrland": ["här", "där", "int", "inte", "så", "mycke", "mycket", "gör", "göra", "va", "eller hur"],
    "skåne": ["här", "där", "inte", "mycke", "gör"],
    "stockholm": ["typ", "liksom", "alltså", "sådär"],
}

SLANG = ["fet", "sjuk", "ball", "naj", "jätte", "skit", "fan", "helvete"]


@register_analyzer("dialect_sensitivity")
class DialectSensitivityAnalyzer(Analyzer):
    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "dialect_sensitivity"

    @property
    def requires(self) -> list[str]:
        return []

    def analyze(self, ctx: AnalysisContext) -> dict[str, Any]:
        if not ctx.segments:
            return {"dialect_risk": "low", "flagged_segments": []}

        flagged = []
        dialect_hits = 0
        slang_hits = 0

        for seg in ctx.segments:
            text = (seg.text or "").lower()
            hits = []

            for dialect, markers in DIALECT_MARKERS.items():
                if any(m in text for m in markers):
                    hits.append(dialect)
                    dialect_hits += 1

            slang_found = [s for s in SLANG if s in text]
            if slang_found:
                slang_hits += len(slang_found)
                hits.append("slang")

            if hits:
                flagged.append({
                    "speaker": getattr(seg, "speaker", None),
                    "start": getattr(seg, "start", 0),
                    "end": getattr(seg, "end", 0),
                    "detected": hits,
                    "confidence_impact": "medium" if len(hits) > 1 else "low",
                    "text_snippet": seg.text[:70] if seg.text else "",
                })

        risk = "high" if dialect_hits > 3 or slang_hits > 2 else ("medium" if flagged else "low")

        return {
            "dialect_risk_level": risk,
            "flagged_segments": flagged,
            "total_dialect_hits": dialect_hits,
            "slang_count": slang_hits,
            "recommendation": "Överväg bättre dialekt-anpassad ASR-modell" if risk == "high" else "Inga större dialekt-problem",
        }
"