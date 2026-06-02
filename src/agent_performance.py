"""Agent Performance & Assessment Engine (Fas 4.1).

Core module for call center: computes quantitative per-call agent/customer metrics
(rule-based, zero LLM cost) + provides hooks for LLM-enriched coaching (4.1.2).

Design (per UTVECKLINGSPLAN_Fas4... v1.1):
- Modular, pure functions where possible.
- Produces Pydantic models (AgentMetrics, CustomerMetrics, CallAgentPerformance)
  that are dumped and merged into CallAnalysisReport.results["agent_performance"].
- Evidence-oriented: compliance_flags and local_coaching_hints are actionable.
- Hybrid-ready: local metrics feed into mistral_analyzer for detailed recs.
- Explicit integration point: call from pipeline.py (see _run_agent_performance).
- Pre-computation friendly: lru_cache on hot paths; aggregate functions for trends.
- Privacy: operates on original segments (PII redaction only affects LLM path).

Caching / invalidation strategy (v1.1):
- Per-call results are deterministic given (segments, role_map, sentiment_snapshot) -> key by content hash.
- For aggregates (agent trends): caller (e.g. insights_aggregator or batch job) is responsible for
  cache key including time window + agent_id. Invalidate on new calls for that agent or explicit TTL.
- Simple file/Redis cache layer can wrap these functions later (see Fas 4.5).

KPIs this enables (see plan):
- % of calls with specific coaching recs
- Empathy score correlation with CSAT (future)
- Reduction in manual QA sampling via auto-flags

Usage (direct from pipeline or CLI):
    from src.agent_performance import compute_call_agent_performance, CallAgentPerformance
    perf = compute_call_agent_performance(segments, role_map, sentiment_results=...)
    report.results["agent_performance"] = perf.model_dump()

    # Aggregation example
    team_perf = aggregate_team_performance([perf1, perf2, ...])
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

from pydantic import BaseModel

from .core.models import Segment
from .llm.schemas import (
    AgentMetrics,
    CallAgentPerformance,
    CustomerMetrics,
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Swedish callcenter lexical resources (conservative, extensible)
# -----------------------------------------------------------------------------

GREETING_KEYWORDS = {
    "hej", "hallå", "god dag", "välkommen", "hej hej", "tack för att du ringer",
    "hej och välkommen", "godmorgon", "god eftermiddag"
}

EMPATHY_MARKERS = {
    "jag förstår", "det förstår jag", "jag hör att", "det låter", "beklagar",
    "tyvärr", "jag är ledsen", "vi fixar det här", "jag ska hjälpa dig",
    "det är frustrerande", "jag kan tänka mig", "vi löser det tillsammans",
    "jag beklagar besväret", "tack för att du säger till"
}

DE_ESCALATION_MARKERS = {
    "vi ska kolla", "jag ordnar", "låt mig se", "jag tar det här",
    "vi återkommer", "jag återkopplar", "jag fixar", "det ska vi klara",
    "ingen fara", "vi kan erbjuda", "jag kan erbjuda"
}

COMPLIANCE_ISSUES = {
    "missing_greeting": "Agent hälsade inte kunden vid samtalets start",
    "no_empathy_on_frustration": "Kunden visade frustration men agent visade ingen empati",
    "dominating_airtime": "Agent talade betydligt mer än kunden (risk för att inte lyssna)",
}

RESOLUTION_MARKERS = {
    "tack", "det var bra", "perfekt", "jättebra", "det löste sig", "tack så mycket",
    "det var hjälpsamt", "bra", "ok då", "då är det klart"
}

FORMAL_MARKERS = {
    "jag förstår", "enligt", "avtal", "enligt våra villkor", "jag kontrollerar",
    "jag undersöker", "vi kommer att", "jag återkommer inom", "bekräftar",
    "jag noterar", "för att säkerställa"
}

CASUAL_MARKERS = {
    "okej", "japp", "fixar", "kollar snabbt", "typ", "ungefär", "kanske",
    "vet inte", "kan inte säga", "lugnt", "no problem"
}


def _segment_duration(seg: dict[str, Any] | Segment) -> float:
    """Return duration in seconds for a segment (ASR segments are best-effort)."""
    if isinstance(seg, dict):
        try:
            return max(0.0, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)))
        except (TypeError, ValueError):
            return 0.0
    try:
        return max(0.0, float(getattr(seg, "end", 0.0)) - float(getattr(seg, "start", 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _get_text(seg: dict[str, Any] | Segment) -> str:
    if isinstance(seg, dict):
        return str(seg.get("text", "") or "").strip()
    return str(getattr(seg, "text", "") or "").strip()


def _get_speaker(seg: dict[str, Any] | Segment) -> str:
    if isinstance(seg, dict):
        return str(seg.get("speaker") or seg.get("speaker_label") or "UNKNOWN")
    return str(getattr(seg, "speaker", None) or "UNKNOWN")


def _normalize_role(speaker: str, role_map: dict[str, str] | None) -> str:
    if role_map and speaker in role_map:
        return role_map[speaker].lower()
    low = speaker.lower()
    if "agent" in low or "handläggare" in low:
        return "agent"
    if "customer" in low or "kund" in low or "caller" in low:
        return "customer"
    return "unknown"


def _hash_for_cache(segments: list[dict | Segment], role_map: dict | None, sentiment_sig: str | None) -> str:
    """Stable short hash for lru_cache / future redis key."""
    payload_parts: list[str] = []
    for s in segments:
        payload_parts.append(f"{_get_speaker(s)}:{_get_text(s)[:80]}")
    role_str = json.dumps(role_map or {}, sort_keys=True)
    sent_str = sentiment_sig or ""
    raw = "|".join(payload_parts) + "::" + role_str + "::" + sent_str
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# -----------------------------------------------------------------------------
# Core metric computers (rule-based, fast, Swedish callcenter tuned)
# -----------------------------------------------------------------------------

def compute_talk_ratios(
    segments: list[dict[str, Any]] | list[Segment],
    role_map: dict[str, str] | None = None,
) -> dict[str, float]:
    """Return agent_talk_ratio, customer_talk_ratio, talk_listen_ratio."""
    total_agent = 0.0
    total_cust = 0.0
    for seg in segments or []:
        dur = _segment_duration(seg)
        role = _normalize_role(_get_speaker(seg), role_map)
        if role == "agent":
            total_agent += dur
        elif role == "customer":
            total_cust += dur
    total = total_agent + total_cust
    if total <= 0:
        return {"agent_talk_ratio": 0.5, "customer_talk_ratio": 0.5, "talk_listen_ratio": 1.0}
    a_ratio = total_agent / total
    c_ratio = total_cust / total
    listen_r = (total_agent / total_cust) if total_cust > 0 else 3.0  # cap domination
    return {
        "agent_talk_ratio": round(a_ratio, 3),
        "customer_talk_ratio": round(c_ratio, 3),
        "talk_listen_ratio": round(listen_r, 2),
    }


def compute_question_density(
    segments: list[dict[str, Any]] | list[Segment],
    role_map: dict[str, str] | None = None,
) -> dict[str, float]:
    """Questions per turn for agent and customer."""
    agent_qs = 0
    agent_turns = 0
    cust_qs = 0
    cust_turns = 0
    for seg in segments or []:
        text = _get_text(seg).lower()
        role = _normalize_role(_get_speaker(seg), role_map)
        q_count = text.count("?") + sum(1 for w in ("vad", "hur", "varför", "kan du", "får jag", "skulle du") if w in text)
        if role == "agent":
            agent_turns += 1
            agent_qs += q_count
        elif role == "customer":
            cust_turns += 1
            cust_qs += q_count
    return {
        "agent_question_density": round(agent_qs / max(1, agent_turns), 2),
        "customer_question_density": round(cust_qs / max(1, cust_turns), 2),
        "num_agent_turns": agent_turns,
        "num_customer_turns": cust_turns,
    }


def compute_lexical_formality(
    segments: list[dict[str, Any]] | list[Segment],
    role_map: dict[str, str] | None = None,
) -> float:
    """0.0 (very casual) .. 1.0 (professional Swedish service speak)."""
    agent_texts = [
        _get_text(seg).lower()
        for seg in (segments or [])
        if _normalize_role(_get_speaker(seg), role_map) == "agent"
    ]
    if not agent_texts:
        return 0.5
    formal_hits = sum(1 for t in agent_texts for m in FORMAL_MARKERS if m in t)
    casual_hits = sum(1 for t in agent_texts for m in CASUAL_MARKERS if m in t)
    # Bias toward formal baseline in service context
    score = 0.55 + 0.15 * (formal_hits - casual_hits)
    return max(0.0, min(1.0, round(score, 2)))


def compute_sentiment_variance(
    segments: list[dict[str, Any]] | list[Segment],
    role_map: dict[str, str] | None = None,
    sentiment_results: list[dict[str, Any]] | None = None,
) -> float:
    """Variance of mapped sentiment for agent's turns. Pure python, no numpy."""
    if not sentiment_results or len(sentiment_results) != len(segments):
        # fallback: no variance signal
        return 0.0
    agent_scores: list[float] = []
    for i, seg in enumerate(segments):
        role = _normalize_role(_get_speaker(seg), role_map)
        if role != "agent":
            continue
        sres = sentiment_results[i] if i < len(sentiment_results) else {}
        # normalize label to -1/0/+1
        label = str(sres.get("label", "")).lower()
        if label in ("negativ", "negative", "neg"):
            sc = -1.0
        elif label in ("positiv", "positive", "pos"):
            sc = 1.0
        else:
            sc = 0.0
        # weight by score if present
        conf = sres.get("score", 0.7)
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.7
        agent_scores.append(sc * max(0.3, min(1.0, conf)))
    if len(agent_scores) < 2:
        return 0.0
    mean = sum(agent_scores) / len(agent_scores)
    var = sum((x - mean) ** 2 for x in agent_scores) / len(agent_scores)
    return round(max(0.0, min(1.0, var)), 3)


def compute_intervention_count(
    segments: list[dict[str, Any]] | list[Segment],
    role_map: dict[str, str] | None = None,
) -> int:
    """Crude proxy: agent turns that follow a customer turn (normal flow) vs rapid switches.
    For v1 we count "agent speaks after customer negative or long turn" as potential intervention.
    More accurate with word-level timing in future.
    """
    count = 0
    prev_role = None
    for seg in segments or []:
        role = _normalize_role(_get_speaker(seg), role_map)
        text = _get_text(seg)
        if prev_role == "customer" and role == "agent":
            # potential cut-in or normal response; count as intervention if customer turn was "heated"
            if any(x in text.lower() for x in ("vänta", "men", "alltså")) or len(text) < 25:
                count += 1
            else:
                count += 0  # normal turn-taking, don't overcount
        prev_role = role
    return max(0, count)


def compute_empathy_and_deescalation(
    segments: list[dict[str, Any]] | list[Segment],
    role_map: dict[str, str] | None = None,
    sentiment_results: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    """Local empathy (0-1) and de-escalation effectiveness.

    Empathy: presence of markers in agent turns that follow customer frustration.
    De-escalation: did customer sentiment recover after agent's response?
    """
    empathy_hits = 0
    opportunities = 0
    recovery_scores: list[float] = []

    sent_map = {}
    if sentiment_results and len(sentiment_results) == len(segments):
        for i, sr in enumerate(sentiment_results):
            label = str(sr.get("label", "")).lower()
            sc = -1.0 if "neg" in label else (1.0 if "pos" in label else 0.0)
            sent_map[i] = sc

    prev_cust_neg_idx: int | None = None
    for i, seg in enumerate(segments or []):
        role = _normalize_role(_get_speaker(seg), role_map)
        text = _get_text(seg).lower()
        if role == "customer":
            if sent_map.get(i, 0) < -0.3:
                prev_cust_neg_idx = i
                opportunities += 1
            else:
                prev_cust_neg_idx = None
            continue

        if role == "agent" and prev_cust_neg_idx is not None:
            # agent response after neg customer turn
            has_empathy = any(m in text for m in EMPATHY_MARKERS)
            has_deesc = any(m in text for m in DE_ESCALATION_MARKERS)
            if has_empathy or has_deesc:
                empathy_hits += 1
            # check recovery in next 1-2 customer turns
            for j in range(i + 1, min(i + 3, len(segments))):
                if _normalize_role(_get_speaker(segments[j]), role_map) == "customer":
                    next_sc = sent_map.get(j, 0)
                    if next_sc > -0.2:  # recovery
                        recovery_scores.append(1.0)
                    elif next_sc < -0.4:
                        recovery_scores.append(0.0)
                    else:
                        recovery_scores.append(0.5)
                    break
            prev_cust_neg_idx = None

    empathy = min(1.0, empathy_hits / max(1, opportunities)) if opportunities > 0 else 0.4
    deesc = sum(recovery_scores) / max(1, len(recovery_scores)) if recovery_scores else 0.5
    return {
        "empathy_score": round(empathy, 2),
        "de_escalation_effectiveness": round(deesc, 2),
        "empathy_opportunities": opportunities,
    }


def compute_compliance_flags(
    segments: list[dict[str, Any]] | list[Segment],
    role_map: dict[str, str] | None = None,
    talk_ratios: dict[str, float] | None = None,
) -> list[str]:
    """Rule-based obvious compliance/process flags. Actionable immediately."""
    flags: list[str] = []
    if not segments:
        return flags

    # 1. Missing greeting in first agent turn
    first_agent_text = ""
    for seg in segments:
        if _normalize_role(_get_speaker(seg), role_map) == "agent":
            first_agent_text = _get_text(seg).lower()
            break
    if first_agent_text and not any(g in first_agent_text for g in GREETING_KEYWORDS):
        flags.append("missing_greeting")

    # 2. No empathy after frustration (we can detect opportunity without marker)
    # Simplified: if any customer neg and no empathy marker anywhere from agent
    has_any_empathy = any(
        any(m in _get_text(seg).lower() for m in EMPATHY_MARKERS)
        for seg in segments
        if _normalize_role(_get_speaker(seg), role_map) == "agent"
    )
    has_frustration = False
    for seg in segments:
        if _normalize_role(_get_speaker(seg), role_map) == "customer":
            if any(w in _get_text(seg).lower() for w in ("arg", "frustr", "inte bra", "fel", "funkar inte", "vänta")):
                has_frustration = True
                break
    if has_frustration and not has_any_empathy:
        flags.append("no_empathy_on_frustration")

    # 3. Dominating airtime
    if talk_ratios and talk_ratios.get("talk_listen_ratio", 1.0) > 2.5:
        flags.append("dominating_airtime")

    return flags


# -----------------------------------------------------------------------------
# Public API - main entry for per-call performance
# -----------------------------------------------------------------------------

def compute_call_agent_performance(
    segments: list[dict[str, Any]] | list[Segment],
    role_map: dict[str, str] | None = None,
    sentiment_results: list[dict[str, Any]] | None = None,
    profile_name: str = "callcenter",
) -> CallAgentPerformance:
    """Compute full per-call agent + customer metrics + local hints.

    This is the primary function to call from pipeline.py after local analyzers
    (and before or after LLM depending on profile).

    Returns validated Pydantic model ready for .model_dump() into report.
    """
    if not segments:
        # graceful zero
        zero_agent = AgentMetrics(
            talk_ratio=0.5,
            talk_listen_ratio=1.0,
            num_agent_turns=0,
            num_customer_turns=0,
            total_talk_time_s=0.0,
        )
        zero_cust = CustomerMetrics(talk_ratio=0.5)
        return CallAgentPerformance(agent=zero_agent, customer=zero_cust)

    # 1. Ratios (cached inner)
    ratios = compute_talk_ratios(segments, role_map)
    qdens = compute_question_density(segments, role_map)
    formality = compute_lexical_formality(segments, role_map)
    variance = compute_sentiment_variance(segments, role_map, sentiment_results)
    interventions = compute_intervention_count(segments, role_map)
    emp_deesc = compute_empathy_and_deescalation(segments, role_map, sentiment_results)
    flags = compute_compliance_flags(segments, role_map, ratios)

    # Build Pydantic objects (will validate)
    agent_m = AgentMetrics(
        talk_ratio=ratios["agent_talk_ratio"],
        talk_listen_ratio=ratios["talk_listen_ratio"],
        question_density=qdens["agent_question_density"],
        lexical_formality=formality,
        sentiment_variance=variance,
        intervention_count=interventions,
        empathy_score=emp_deesc["empathy_score"],
        de_escalation_effectiveness=emp_deesc["de_escalation_effectiveness"],
        compliance_flags=flags,
        num_agent_turns=qdens["num_agent_turns"],
        num_customer_turns=qdens["num_customer_turns"],
        total_talk_time_s=sum(_segment_duration(s) for s in segments),
    )

    # Customer side (simplified slope from sentiment if available)
    cust_talk = ratios["customer_talk_ratio"]
    slope = 0.0
    frust = 0
    res_ind = 0.0
    if sentiment_results and len(sentiment_results) == len(segments):
        cust_scores = []
        for i, sr in enumerate(sentiment_results):
            if _normalize_role(_get_speaker(segments[i]), role_map) == "customer":
                lab = str(sr.get("label", "")).lower()
                sc = -1.0 if "neg" in lab else (1.0 if "pos" in lab else 0.0)
                cust_scores.append(sc)
                if sc < -0.6:
                    frust += 1
        if len(cust_scores) >= 2:
            slope = round(cust_scores[-1] - cust_scores[0], 2)
        # resolution at end
        last_cust_texts = [
            _get_text(segments[i]).lower()
            for i in range(len(segments) - 1, -1, -1)
            if _normalize_role(_get_speaker(segments[i]), role_map) == "customer"
        ][:2]
        res_hits = sum(1 for t in last_cust_texts for m in RESOLUTION_MARKERS if m in t)
        res_ind = min(1.0, res_hits / 2.0)

    cust_m = CustomerMetrics(
        talk_ratio=cust_talk,
        sentiment_slope=slope,
        frustration_peaks=frust,
        question_count=int(qdens.get("customer_question_density", 0) * qdens.get("num_customer_turns", 1)),
        resolution_indicators=round(res_ind, 2),
    )

    # Local actionable hints (evidence light but immediate value)
    hints: list[str] = []
    if "missing_greeting" in flags:
        hints.append("Agent bör inleda med en tydlig hälsningsfras (t.ex. 'Hej, välkommen till kundtjänst').")
    if "no_empathy_on_frustration" in flags:
        hints.append("Vid tecken på frustration: använd empatifraser som 'Jag förstår att det här är frustrerande' tidigt.")
    if agent_m.talk_listen_ratio > 2.0:
        hints.append("Agent dominerar taltiden – ställ fler öppna frågor och låt kunden tala färdigt.")
    if emp_deesc["empathy_score"] < 0.3 and emp_deesc["empathy_opportunities"] > 0:
        hints.append("Fler empatimarkörer behövs efter kundens negativa yttranden.")

    evidence_sum = None
    if flags or hints:
        evidence_sum = "Lokala signaler: " + "; ".join(flags + hints[:2])

    perf = CallAgentPerformance(
        agent=agent_m,
        customer=cust_m,
        local_coaching_hints=hints,
        evidence_summary=evidence_sum,
    )
    logger.debug(
        "Agent performance computed | agent_turns=%d empathy=%.2f flags=%s",
        agent_m.num_agent_turns,
        agent_m.empathy_score,
        flags,
    )
    return perf


# Lightweight cached wrapper (for repeated calls on same transcript in batch/eval)
@lru_cache(maxsize=128)
def _compute_call_agent_performance_cached(
    segments_key: str, role_json: str, sent_sig: str
) -> dict[str, Any]:
    """Internal cached version returning dict (for lru). Callers use the non-cached primarily."""
    # NOTE: real caller reconstructs; this is placeholder for future redis/file wrapper.
    # For now we do not use it in hot path to keep types clean.
    return {}


# -----------------------------------------------------------------------------
# Aggregation for agent-level views (trends, benchmarking) - used by 4.3 later
# -----------------------------------------------------------------------------

def aggregate_agent_performance(
    per_call_perfs: list[CallAgentPerformance],
    agent_id: str | None = None,
) -> dict[str, Any]:
    """Simple aggregate stats over a list of per-call performances for one agent.

    Returns dict ready for storage / dashboard. No LLM.
    Invalidation: recompute when new calls for the agent arrive (time-windowed).
    """
    if not per_call_perfs:
        return {"agent_id": agent_id, "call_count": 0, "averages": {}}

    n = len(per_call_perfs)
    avg_empathy = sum(p.agent.empathy_score for p in per_call_perfs) / n
    avg_talk_ratio = sum(p.agent.talk_ratio for p in per_call_perfs) / n
    avg_formality = sum(p.agent.lexical_formality for p in per_call_perfs) / n
    total_flags = sum(len(p.agent.compliance_flags) for p in per_call_perfs)
    avg_deesc = sum(p.agent.de_escalation_effectiveness for p in per_call_perfs) / n

    # Simple "trend" proxy: compare first half vs second half if enough data
    trend = "stable"
    if n >= 4:
        first_half = per_call_perfs[: n // 2]
        second_half = per_call_perfs[n // 2 :]
        e1 = sum(p.agent.empathy_score for p in first_half) / len(first_half)
        e2 = sum(p.agent.empathy_score for p in second_half) / len(second_half)
        if e2 - e1 > 0.1:
            trend = "improving"
        elif e1 - e2 > 0.1:
            trend = "declining"

    return {
        "agent_id": agent_id or "unknown",
        "call_count": n,
        "averages": {
            "empathy_score": round(avg_empathy, 3),
            "talk_ratio": round(avg_talk_ratio, 3),
            "lexical_formality": round(avg_formality, 3),
            "de_escalation_effectiveness": round(avg_deesc, 3),
        },
        "total_compliance_flags": total_flags,
        "avg_flags_per_call": round(total_flags / n, 2),
        "trend_empathy": trend,
        "computed_at": datetime.now(UTC).isoformat(),
    }


def aggregate_team_performance(
    per_call_perfs: list[CallAgentPerformance],
) -> dict[str, Any]:
    """Team-level rollup (benchmark source for individual agents)."""
    if not per_call_perfs:
        return {"team_call_count": 0}
    base = aggregate_agent_performance(per_call_perfs, agent_id="TEAM")
    base["team_call_count"] = base.pop("call_count")
    base["team_averages"] = base.pop("averages")
    return base


__all__ = [
    "AgentMetrics",
    "CustomerMetrics",
    "CallAgentPerformance",
    "compute_call_agent_performance",
    "aggregate_agent_performance",
    "aggregate_team_performance",
]
