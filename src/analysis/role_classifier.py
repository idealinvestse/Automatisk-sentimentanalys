"""Speaker Role Inference (Task 2.2) + extended features for Fas 4.1.1.

Extended with talk_ratio, lexical_formality, question_density, sentiment_variance (if avail),
intervention_count per the UTVECKLINGSPLAN_Fas4 v1.1.

Heuristics for the 4.1.1 features are delegated to the pure functions in agent_performance
(single source of truth, avoids duplication while satisfying the "utöka role_classifier" requirement).

Returns richer structure for downstream (agent_performance.py produces the authoritative
Pydantic AgentMetrics/CustomerMetrics/CallAgentPerformance). Backward compat for
role_map = results.get("role") (or role_res.get("roles")) is preserved.

Integration note: pipeline.py and mistral_analyzer.py now do:
    role_res = results.get("role") or {}
    role_map = role_res.get("roles", role_res) if isinstance(role_res, dict) else {}
"""

from __future__ import annotations

import logging
from typing import Any

from ..core.models import AnalysisContext
from .base import Analyzer
from .registry import register_analyzer

logger = logging.getLogger(__name__)


# Delegate to shared pure heuristics in agent_performance (single source of truth for 4.1.1 features).
# This eliminates duplication while keeping role_classifier lightweight and early in the analyzer graph.
# Role still owns the basic speaker->role heuristic.
try:
    from ..agent_performance import (
        compute_intervention_count,
        compute_lexical_formality,
        compute_question_density,
        compute_sentiment_variance,
        compute_talk_ratios,
    )
except Exception:  # fallback if circular/import during very early load (should not happen)
    compute_talk_ratios = None
    compute_question_density = None
    compute_lexical_formality = None
    compute_sentiment_variance = None
    compute_intervention_count = None


def _role_from_speaker(sp: str) -> str:
    low = (sp or "").lower()
    if "agent" in low or "handlägg" in low:
        return "agent"
    if "customer" in low or "kund" in low:
        return "customer"
    return "unknown"


@register_analyzer("role")
class RoleAnalyzer(Analyzer):
    @property
    def name(self) -> str:
        return "role"

    @property
    def requires(self) -> list[str]:
        return ["sentiment"]  # sentiment_variance needs aligned sentiment results

    def analyze(self, ctx: AnalysisContext) -> dict[str, Any]:
        """Return role map + extended Fas 4.1 features.

        Structure:
            {
              "roles": {"SPEAKER_0": "agent", ...},  # backward compat primary
              "talk_ratios": {"agent": 0.62, "customer": 0.38},
              "question_density": {"agent": 0.8, ...},
              "lexical_formality": 0.72,
              "intervention_count": 1,
              "sentiment_variance": 0.12,  # only if sentiment in ctx.results
              ...
            }
        Callers that only need map do: role_map = d.get("roles", d) if isinstance...
        """
        segments = ctx.segments or []
        speakers = []
        seen = set()
        for s in segments:
            sp = getattr(s, "speaker", None) or (s.get("speaker") if isinstance(s, dict) else None)
            if sp and sp not in seen:
                seen.add(sp)
                speakers.append(sp)
        speakers = sorted(seen)

        roles: dict[str, str] = {}
        if len(speakers) >= 2:
            # Heuristic improved: first speaker often agent in inbound callcenter
            roles[speakers[0]] = "agent"
            roles[speakers[1]] = "customer"
        else:
            for sp in speakers:
                roles[sp] = _role_from_speaker(sp) or "unknown"

        # --- Extended Fas 4.1 features (delegated to shared logic in agent_performance to avoid dupe) ---
        # Use the full compute_* when available (they handle dict/Segment, role_map etc.)
        if compute_talk_ratios:
            ratios = compute_talk_ratios(segments, roles)
            a_talk = ratios.get("agent_talk_ratio", 0.5)
            c_talk = ratios.get("customer_talk_ratio", 0.5)
            talk_listen = ratios.get("talk_listen_ratio", 1.0)
        else:
            a_talk = c_talk = 0.5
            talk_listen = 1.0

        if compute_question_density:
            qd = compute_question_density(segments, roles)
            qdens_a = qd.get("agent_question_density", 0.0)
            qdens_c = qd.get("customer_question_density", 0.0)
            a_turns = qd.get("num_agent_turns", 0)
            c_turns = qd.get("num_customer_turns", 0)
        else:
            qdens_a = qdens_c = 0.0
            a_turns = c_turns = 0

        form_score = compute_lexical_formality(segments, roles) if compute_lexical_formality else 0.5

        inter = compute_intervention_count(segments, roles) if compute_intervention_count else 0

        # sentiment_variance (needs prior sentiment results)
        var = 0.0
        sent = ctx.results.get("sentiment") if hasattr(ctx, "results") else None
        if compute_sentiment_variance and sent and len(sent) == len(segments):
            var = compute_sentiment_variance(segments, roles, sent)

        out: dict[str, Any] = {
            "roles": roles,  # primary for role_map extraction
            "talk_ratios": {"agent": a_talk, "customer": c_talk, "talk_listen_ratio": talk_listen},
            "question_density": {"agent": qdens_a, "customer": qdens_c},
            "lexical_formality": form_score,
            "intervention_count": inter,
            "sentiment_variance": var,
            "num_agent_turns": a_turns,
            "num_customer_turns": c_turns,
        }
        return out
