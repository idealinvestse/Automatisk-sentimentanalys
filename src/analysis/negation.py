"""Negation detection analyzer adapter for the registry dependency graph."""

from __future__ import annotations

import logging
from typing import Any

from ..core.models import AnalysisContext
from ..negation import detect_negation, detect_negation_with_position
from .base import Analyzer
from .evidence import evidence_to_dict, make_evidence_span
from .registry import register_analyzer

logger = logging.getLogger(__name__)


@register_analyzer("negation")
class NegationAnalyzer(Analyzer):
    """Detect Swedish negation markers per segment (feeds empathy and sentiment QA)."""

    @property
    def name(self) -> str:
        return "negation"

    @property
    def requires(self) -> list[str]:
        return []

    def analyze(self, ctx: AnalysisContext) -> list[dict[str, Any]]:
        if not ctx.segments:
            return []

        out: list[dict[str, Any]] = []
        for idx, seg in enumerate(ctx.segments):
            text = seg.text or ""
            positions = detect_negation_with_position(text)
            evidence_spans: list[dict[str, Any]] = []
            for pos_idx, ntype in positions[:3]:
                snippet = text[max(0, pos_idx - 8) : pos_idx + 24].strip()
                if snippet:
                    evidence_spans.append(
                        evidence_to_dict(
                            make_evidence_span(
                                snippet,
                                speaker_role=getattr(seg, "speaker", None),
                                turn_index=idx,
                                segment_id=idx,
                                start=getattr(seg, "start", None),
                                end=getattr(seg, "end", None),
                            )
                        )
                    )
            out.append(
                {
                    "has_negation": bool(positions) or detect_negation(text),
                    "negation_count": len(positions),
                    "positions": [{"index": pos_idx, "type": ntype} for pos_idx, ntype in positions],
                    "speaker": getattr(seg, "speaker", None),
                    "evidence_spans": evidence_spans,
                }
            )
        return out
