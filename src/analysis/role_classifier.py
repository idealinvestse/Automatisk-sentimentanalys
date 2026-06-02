"""Speaker Role Inference (Task 2.2) + extended features for Fas 4.1.1.

Extended with talk_ratio, lexical_formality, question_density, sentiment_variance (if avail),
intervention_count per the UTVECKLINGSPLAN_Fas4 v1.1.

Returns richer structure for downstream (agent_performance.py does the authoritative
full AgentMetrics/CustomerMetrics Pydantic models). This keeps backward compat for
role_map = results.get("role") while providing "role_features" etc.

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


# Light-weight Swedish markers (duplicated small set to keep role independent of agent_performance
# for early pipeline stages; authoritative logic lives in src/agent_performance.py).
_GREET = {"hej", "hallå", "god dag", "välkommen"}
_FORM = {"jag förstår", "beklagar", "tyvärr", "jag kontrollerar", "vi ska"}
_CAS = {"okej", "japp", "fixar", "kollar"}


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
        return []  # can use diarization results if present in ctx; sentiment optional for variance

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

        # --- Extended Fas 4.1 features (light, no heavy deps) ---
        total_dur_agent = 0.0
        total_dur_cust = 0.0
        for s in segments:
            dur = 0.0
            try:
                dur = max(0.0, float(getattr(s, "end", 0)) - float(getattr(s, "start", 0)))
            except Exception:
                dur = 0.0
            sp = getattr(s, "speaker", None) or (s.get("speaker") if isinstance(s, dict) else None)
            r = roles.get(sp, _role_from_speaker(sp or ""))
            if r == "agent":
                total_dur_agent += dur
            elif r == "customer":
                total_dur_cust += dur

        tot = total_dur_agent + total_dur_cust
        a_talk = round(total_dur_agent / tot, 3) if tot > 0 else 0.5
        c_talk = round(total_dur_cust / tot, 3) if tot > 0 else 0.5
        talk_listen = round(total_dur_agent / total_dur_cust, 2) if total_dur_cust > 0 else 2.0

        # question density (simple)
        a_qs = 0
        c_qs = 0
        a_turns = 0
        c_turns = 0
        for s in segments:
            txt = (getattr(s, "text", "") or (s.get("text", "") if isinstance(s, dict) else "")).lower()
            sp = getattr(s, "speaker", None) or (s.get("speaker") if isinstance(s, dict) else None)
            r = roles.get(sp, _role_from_speaker(sp or ""))
            qs = txt.count("?")
            if r == "agent":
                a_turns += 1
                a_qs += qs
            elif r == "customer":
                c_turns += 1
                c_qs += qs
        qdens_a = round(a_qs / max(1, a_turns), 2)
        qdens_c = round(c_qs / max(1, c_turns), 2)

        # lexical_formality (agent only)
        formal = 0
        casual = 0
        for s in segments:
            txt = (getattr(s, "text", "") or (s.get("text", "") if isinstance(s, dict) else "")).lower()
            sp = getattr(s, "speaker", None) or (s.get("speaker") if isinstance(s, dict) else None)
            if roles.get(sp, _role_from_speaker(sp or "")) != "agent":
                continue
            formal += sum(1 for m in _FORM if m in txt)
            casual += sum(1 for m in _CAS if m in txt)
        form_score = max(0.0, min(1.0, round(0.55 + 0.12 * (formal - casual), 2)))

        # intervention proxy (agent follows customer)
        inter = 0
        prev_r = None
        for s in segments:
            sp = getattr(s, "speaker", None) or (s.get("speaker") if isinstance(s, dict) else None)
            r = roles.get(sp, _role_from_speaker(sp or ""))
            if prev_r == "customer" and r == "agent":
                inter += 1
            prev_r = r

        # sentiment_variance if available from prior analyzer (ctx.results)
        var = 0.0
        sent = ctx.results.get("sentiment") if hasattr(ctx, "results") else None
        if isinstance(sent, list) and len(sent) == len(segments) and a_turns > 1:
            ag_scores: list[float] = []
            for i, s in enumerate(segments):
                sp = getattr(s, "speaker", None) or (s.get("speaker") if isinstance(s, dict) else None)
                if roles.get(sp, _role_from_speaker(sp or "")) != "agent":
                    continue
                sr = sent[i] if i < len(sent) else {}
                lab = str(sr.get("label", "")).lower() if isinstance(sr, dict) else ""
                sc = -1.0 if "neg" in lab else (1.0 if "pos" in lab else 0.0)
                ag_scores.append(sc)
            if len(ag_scores) >= 2:
                m = sum(ag_scores) / len(ag_scores)
                var = round(sum((x - m) ** 2 for x in ag_scores) / len(ag_scores), 3)

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
