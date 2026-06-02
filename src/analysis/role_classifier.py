"""Speaker Role Inference (Task 2.2).

Simple heuristic: agent vs customer based on talk ratio, question density, formality.
"""

from __future__ import annotations

import logging
from typing import Any

from ..core.models import AnalysisContext
from .base import Analyzer
from .registry import register_analyzer

logger = logging.getLogger(__name__)


@register_analyzer("role")
class RoleAnalyzer(Analyzer):
    @property
    def name(self) -> str:
        return "role"

    @property
    def requires(self) -> list[str]:
        return []  # can use diarization results if present in ctx

    def analyze(self, ctx: AnalysisContext) -> dict[str, str]:
        # Very basic: assume first speaker (often agent in callcenter) vs second
        roles = {}
        speakers = set()
        for s in ctx.segments or []:
            if s.speaker:
                speakers.add(s.speaker)
        speakers = sorted(speakers)
        if len(speakers) >= 2:
            roles[speakers[0]] = "agent"
            roles[speakers[1]] = "customer"
        else:
            for sp in speakers:
                roles[sp] = "unknown"
        return roles
