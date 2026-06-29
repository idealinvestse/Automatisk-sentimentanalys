"""DialectSensitivityAnalyzer.

Detects regional Swedish dialect markers and slang that may affect ASR or analysis accuracy.
Uses distinctive dialect forms only (not common words like "här", "där", "inte").
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ..core.models import AnalysisContext
from .base import Analyzer
from .registry import register_analyzer

logger = logging.getLogger(__name__)

# Distinctive regional/dialect forms — avoid high-frequency standard Swedish words
DIALECT_MARKERS: dict[str, list[str]] = {
    "dalarna/norrland": ["mycke", "nåt", "nä", "dom", "int'e", "förresten", "jö"],
    "skåne": ["rälig", "ingen fara", "jösses", "mycke"],
    "gotland": ["dej", "rejs", "nån"],
    "stockholm_slang": ["sådär", "asså"],
}

SLANG = ["fet", "sjuk", "ball", "naj", "skit", "fan", "helvete"]

_DIALECT_REGEX = {
    region: re.compile(r"\b(" + "|".join(re.escape(m) for m in markers) + r")\b", re.IGNORECASE)
    for region, markers in DIALECT_MARKERS.items()
}


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
            return {
                "dialect_risk_level": "low",
                "flagged_segments": [],
                "total_dialect_hits": 0,
                "slang_count": 0,
                "recommendation": "Inga större dialekt-problem",
            }

        flagged = []
        dialect_hits = 0
        slang_hits = 0

        for seg in ctx.segments:
            text = (seg.text or "").lower()
            hits: list[str] = []

            for region, regex in _DIALECT_REGEX.items():
                if regex.search(text):
                    hits.append(region)
                    dialect_hits += 1

            slang_found = [s for s in SLANG if re.search(rf"\b{re.escape(s)}\b", text)]
            if slang_found:
                slang_hits += len(slang_found)
                hits.append("slang")

            if hits:
                flagged.append(
                    {
                        "speaker": getattr(seg, "speaker", None),
                        "start": getattr(seg, "start", 0),
                        "end": getattr(seg, "end", 0),
                        "detected": hits,
                        "confidence_impact": "medium" if len(hits) > 1 else "low",
                        "text_snippet": seg.text[:70] if seg.text else "",
                    }
                )

        risk = "high" if dialect_hits > 3 or slang_hits > 2 else ("medium" if flagged else "low")

        return {
            "dialect_risk_level": risk,
            "flagged_segments": flagged,
            "total_dialect_hits": dialect_hits,
            "slang_count": slang_hits,
            "recommendation": (
                "Överväg bättre dialekt-anpassad ASR-modell"
                if risk == "high"
                else "Inga större dialekt-problem"
            ),
        }
