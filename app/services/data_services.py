"""Data services for the NiceGUI Call Center Dashboard MVP (Fas 5.0).

This module provides:
- Realistic canned Swedish call center demo transcripts (4-6 short conversations).
- generate_demo_reports(...) that runs the *real* CallAnalysisPipeline (lru_cache)
  (profile="callcenter") on them and returns serializable dicts (r.to_dict()).
- Filter, KPI, summary and enrichment helpers that power the enhanced dashboard + Call Detail View.
- Support for user-uploaded full CallAnalysisReport JSON (single or list).
- Graceful fallbacks when Fas3/4 fields (llm, qa, agent_performance, alerts, results) are absent.

Design notes for migratability (per plan):
- All helpers take/return plain serializable data (dicts, lists, primitives).
- No UI state here (pure functions + cached).
- Future React: these become API response mappers / React Query hooks or utils.
- "Always show evidence" from backend (spans, quotes, local or llm) is respected by callers.
- Performance: heavy pipeline runs are cached by st.cache_data; transcripts are small & deterministic.

Usage in dashboard:
    from app.services.data_services import (
        get_demo_transcripts, generate_demo_reports,
        extract_call_summary, get_overall_sentiment, compute_kpis,
        filter_reports, enrich_segments_with_sentiment,
    )
"""

from __future__ import annotations

import hashlib
import json
import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CANNED REALISTIC SWEDISH CALL CENTER DEMO TRANSCRIPTS
# 5 short conversations (8-15 turns) covering:
#   - Positive resolution (happy path)
#   - Negative/escalation (poor empathy, risk)
#   - Complex technical + near-miss compliance
#   - Billing dispute with root cause + actionable
#   - Successful de-escalation + upsell light
#
# Speakers use "Agent" / "Kund" (standardized; pipeline normalizes).
# start/end are approximate seconds for timeline (normalized later in UI).
# These are designed so that local + LLM (when enabled) produce rich Fas3/4 evidence.
# ---------------------------------------------------------------------------

DEMO_TRANSCRIPTS: list[dict[str, Any]] = [
    {
        "id": "CALL-001",
        "title": "Faktura fel – lyckad upplösning",
        "meta": {"agent": "Agent-Anna", "duration_s": 420, "category": "billing"},
        "segments": [
            {"start": 0.0, "end": 8.0, "text": "Hej, jag heter Anna på kundtjänst, hur kan jag hjälpa dig idag?", "speaker": "Agent"},
            {"start": 8.0, "end": 18.0, "text": "Hej Anna, jag har fått en faktura på 890 kr som jag inte förstår. Det står att jag har ringt internationellt men det har jag inte.", "speaker": "Kund"},
            {"start": 18.0, "end": 32.0, "text": "Tack för att du ringer in. Jag förstår att det känns frustrerande. Kan jag få ditt kundnummer eller personnummer så kollar jag upp det direkt?", "speaker": "Agent"},
            {"start": 32.0, "end": 42.0, "text": "Ja, det är 19851203-1234. Och jag har aldrig ringt utomlands, jag lovar.", "speaker": "Kund"},
            {"start": 42.0, "end": 65.0, "text": "Tack, jag ser nu i systemet att det finns en debitering från ett samtal till +49 den 12 maj. Men jag ser också att du har ett abonnemang som inkluderar EU-samtal. Det verkar som en felkodning i faktureringssystemet. Jag krediterar 890 kr nu direkt och skickar en rättad faktura.", "speaker": "Agent"},
            {"start": 65.0, "end": 75.0, "text": "Oj, tack! Det var snabbt. Hur lång tid tar det innan det syns på kontot?", "speaker": "Kund"},
            {"start": 75.0, "end": 88.0, "text": "Det syns på nästa faktura eller som kredit inom 3-5 vardagar. Jag lägger också en notering så att det inte händer igen. Är det något mer jag kan hjälpa dig med idag?", "speaker": "Agent"},
            {"start": 88.0, "end": 95.0, "text": "Nej, det var allt. Tack för hjälpen, Anna – du var jättebra!", "speaker": "Kund"},
            {"start": 95.0, "end": 102.0, "text": "Tack själv, ha en bra dag!", "speaker": "Agent"},
        ],
    },
    {
        "id": "CALL-002",
        "title": "Lång väntetid + arg kund – eskaleringsrisk",
        "meta": {"agent": "Agent-Bengt", "duration_s": 310, "category": "complaint"},
        "segments": [
            {"start": 0.0, "end": 5.0, "text": "Ja hallå? Jag har väntat i 45 minuter i kön!", "speaker": "Kund"},
            {"start": 5.0, "end": 12.0, "text": "Hej, tack för att du väntar. Mitt namn är Bengt. Vad gäller ditt ärende?", "speaker": "Agent"},
            {"start": 12.0, "end": 25.0, "text": "Jag ringde för att säga upp mitt abonnemang för två veckor sedan och jag har fortfarande inte fått bekräftelse. Och nu kommer en ny faktura ändå! Detta är skandal!", "speaker": "Kund"},
            {"start": 25.0, "end": 35.0, "text": "Okej, jag förstår att du är upprörd. Men jag behöver ditt kundnummer för att kunna titta.", "speaker": "Agent"},
            {"start": 35.0, "end": 45.0, "text": "Jag har redan gett det i kön! Varför kan ni inte ha koll? Jag vill tala med en chef nu!", "speaker": "Kund"},
            {"start": 45.0, "end": 58.0, "text": "Jag kan inte koppla dig till chef direkt. Låt mig först kolla status på uppsägningen. Kan du upprepa kundnumret?", "speaker": "Agent"},
            {"start": 58.0, "end": 68.0, "text": "19851203-1234. Och jag vill ha skriftlig bekräftelse inom 24 timmar annars kontaktar jag Konsumentverket och ARN!", "speaker": "Kund"},
            {"start": 68.0, "end": 82.0, "text": "Okej, jag ser att uppsägningen registrerades den 14 maj men bekräftelsen gick inte iväg pga tekniskt fel. Jag skickar den nu manuellt och krediterar fakturan. Men jag kan tyvärr inte göra mer idag.", "speaker": "Agent"},
            {"start": 82.0, "end": 90.0, "text": "Det här duger inte. Jag är så less på er. Ni hör av er.", "speaker": "Kund"},
            {"start": 90.0, "end": 95.0, "text": "Tack för samtalet.", "speaker": "Agent"},
        ],
    },
    {
        "id": "CALL-003",
        "title": "Tekniskt fel + compliance near-miss (QA-flagg)",
        "meta": {"agent": "Agent-Cecilia", "duration_s": 480, "category": "tech_support"},
        "segments": [
            {"start": 0.0, "end": 4.0, "text": "Tjenare, det är Cecilia på support.", "speaker": "Agent"},
            {"start": 4.0, "end": 15.0, "text": "Hej, min router har varit nere hela helgen. Jag kan inte jobba. Jag har ringt tidigare och fick löfte om att någon skulle komma ut men ingenting har hänt.", "speaker": "Kund"},
            {"start": 15.0, "end": 22.0, "text": "Okej, tråkigt att höra. Har du provat att starta om routern?", "speaker": "Agent"},
            {"start": 22.0, "end": 30.0, "text": "Ja, tre gånger! Och jag har bytt sladd. Det är ert fel, inte mitt.", "speaker": "Kund"},
            {"start": 30.0, "end": 45.0, "text": "Förstår. Jag kollar i systemet – din linje visar röd sedan fredag. Jag bokar en tekniker till imorgon mellan 8-12. Bekräftar du adressen Storgatan 12?", "speaker": "Agent"},
            {"start": 45.0, "end": 52.0, "text": "Ja, det stämmer. Men jag vill ha kompensation för stilleståndet. Jag har förlorat jobbintäkter.", "speaker": "Kund"},
            {"start": 52.0, "end": 68.0, "text": "Vi har tyvärr ingen policy för det just nu. Men jag kan ge dig 50 kr rabatt på nästa faktura. Är det okej?", "speaker": "Agent"},
            {"start": 68.0, "end": 78.0, "text": "50 kr? Det är ju ingenting. Ni har förstört min helg. Jag vill ha minst 300 kr eller så lämnar jag er.", "speaker": "Kund"},
            {"start": 78.0, "end": 92.0, "text": "Låt mig se vad jag kan göra... Okej, jag lägger in 200 kr goodwill-kredit manuellt. Och tekniker imorgon. Tack för tålamodet.", "speaker": "Agent"},
            {"start": 92.0, "end": 100.0, "text": "Okej, det får duga. Men se till att det blir rätt denna gången.", "speaker": "Kund"},
            {"start": 100.0, "end": 108.0, "text": "Absolut. Ha en bra dag.", "speaker": "Agent"},
        ],
    },
    {
        "id": "CALL-004",
        "title": "Betalningsproblem + root cause (LLM-berikad)",
        "meta": {"agent": "Agent-Daniel", "duration_s": 390, "category": "billing"},
        "segments": [
            {"start": 0.0, "end": 6.0, "text": "Hej, Daniel här. Vad kan jag stå till tjänst med?", "speaker": "Agent"},
            {"start": 6.0, "end": 18.0, "text": "Jag har fått påminnelse om obetald faktura men jag betalade den förra månaden. Varför kommer det här?", "speaker": "Kund"},
            {"start": 18.0, "end": 30.0, "text": "Låt mig kolla. Jag ser att betalningen från 3 april inte har matchats mot rätt faktura i systemet. Det är ett känt problem just nu med vår bankkoppling.", "speaker": "Agent"},
            {"start": 30.0, "end": 40.0, "text": "Men jag har kvitto! Jag kan inte ha det här hängande över mig. Det påverkar min kreditvärdighet.", "speaker": "Kund"},
            {"start": 40.0, "end": 55.0, "text": "Jag beklagar verkligen. Jag markerar fakturan som betald manuellt nu och lägger en spärr så att inga fler påminnelser går ut. Jag skickar också bekräftelse till din e-post.", "speaker": "Agent"},
            {"start": 55.0, "end": 65.0, "text": "Okej... Men hur kunde det bli så här? Har ni inte koll på era system?", "speaker": "Kund"},
            {"start": 65.0, "end": 78.0, "text": "Det är ett internt IT-problem som vår leverantör håller på att fixa. Vi har haft flera fall den här veckan. Jag lägger en intern incidentrapport så att det inte drabbar fler.", "speaker": "Agent"},
            {"start": 78.0, "end": 88.0, "text": "Tack. Jag hoppas det löser sig fort. Annars byter jag operatör.", "speaker": "Kund"},
            {"start": 88.0, "end": 95.0, "text": "Förstår. Är det något annat jag kan hjälpa till med medan vi har kontakt?", "speaker": "Agent"},
        ],
    },
    {
        "id": "CALL-005",
        "title": "De-eskalering + lätt upsell (positiv vändning)",
        "meta": {"agent": "Agent-Erika", "duration_s": 275, "category": "retention"},
        "segments": [
            {"start": 0.0, "end": 7.0, "text": "Hej, det är Erika. Jag såg att du ringde angående ditt abonnemang.", "speaker": "Agent"},
            {"start": 7.0, "end": 18.0, "text": "Ja, jag funderar på att säga upp. Priset har gått upp och jag använder det knappt längre.", "speaker": "Kund"},
            {"start": 18.0, "end": 28.0, "text": "Jag förstår. Många upplever samma sak just nu. Får jag fråga vad du använder mest – mobil eller bredband?", "speaker": "Agent"},
            {"start": 28.0, "end": 35.0, "text": "Främst mobilen. Bredbandet har jag via jobbet.", "speaker": "Kund"},
            {"start": 35.0, "end": 48.0, "text": "Perfekt. Vi har just nu ett erbjudande där du kan behålla mobilt bredband + 100 GB för 199 kr/mån i 6 månader om du behåller abonnemanget. Det är 30 % lägre än nuvarande pris.", "speaker": "Agent"},
            {"start": 48.0, "end": 58.0, "text": "Hmm, 199 låter bättre. Men jag vill inte bindas i 24 månader igen.", "speaker": "Kund"},
            {"start": 58.0, "end": 70.0, "text": "Ingen bindningstid på det här erbjudandet. Du kan säga upp när som helst efter 6 månader. Vill du att jag aktiverar det nu?", "speaker": "Agent"},
            {"start": 70.0, "end": 78.0, "text": "Okej, kör på. Men bara om det verkligen blir 199.", "speaker": "Kund"},
            {"start": 78.0, "end": 88.0, "text": "Klart det blir. Jag aktiverar det nu och skickar bekräftelse. Tack för att du stannar hos oss – uppskattas!", "speaker": "Agent"},
            {"start": 88.0, "end": 93.0, "text": "Tack själv. Hej då.", "speaker": "Kund"},
        ],
    },
]


def get_demo_transcripts() -> list[dict[str, Any]]:
    """Return a copy of the canned demo transcripts (safe to mutate by caller)."""
    return [t.copy() for t in DEMO_TRANSCRIPTS]


def _transcript_hash(transcripts: list[dict]) -> str:
    """Stable short hash for cache key / logging (based on ids + segment count + first text)."""
    h = hashlib.md5()
    for t in transcripts:
        h.update(t.get("id", "").encode())
        h.update(str(len(t.get("segments", []))).encode())
        segs = t.get("segments", [])
        if segs:
            h.update(segs[0].get("text", "")[:30].encode())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------------------
# CORE CACHED GENERATOR – runs REAL pipeline (the heart of "use real pipeline data")
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def generate_demo_reports(
    use_llm: bool = False,
    profile: str = "callcenter",
    force_refresh: bool = False,  # noqa: ARG001 – kept for API compatibility
    llm_api_key: str | None = None,
) -> tuple[dict, ...]:
    """Run CallAnalysisPipeline on the 5 canned transcripts and return list of report.to_dict().

    Each returned dict is a full serializable CallAnalysisReport (with .results, .llm, .segments etc).
    Enriched with 'call_id' and 'title' for UI convenience.

    IMPORTANT:
    - Always uses profile="callcenter" by default (enables Fas4 features + LLM heuristics).
    - use_llm=True will attempt Mistral holistic (requires OPENROUTER_API_KEY env var or configs/openrouter.key for dev).
    - llm_api_key can be passed to override (from dashboard UI for example).
    - lru_cache ensures we do NOT re-run full pipeline on every dashboard reload.
    - Hash key includes use_llm + profile + transcript content hash.

    Returns:
        list[dict] – each is report.to_dict() + call_id/title/meta
    """
    transcripts = get_demo_transcripts()
    thash = _transcript_hash(transcripts)

    # Import inside to avoid circular / heavy import at module load
    try:
        from src.pipeline import CallAnalysisPipeline
    except Exception as import_err:
        logger.warning("Kunde inte importera pipeline: %s. Använder fallback.", import_err)
        return tuple(_generate_fallback_reports(transcripts))

    reports: list[dict] = []
    for t in transcripts:
        try:
            pipe = CallAnalysisPipeline(
                profile=profile,
                use_mistral_llm=use_llm,
                deep_analysis=use_llm,
                llm_api_key=llm_api_key,
            )
            # analyze_segments expects list[dict] with at least "text" (+ optional speaker/start/end)
            report = pipe.analyze_segments(t["segments"])
            rdict: dict = report.to_dict()
        except Exception as run_err:
            logger.warning("Pipeline-fel på %s: %s", t["id"], run_err)
            # Minimal fallback for this one
            rdict = {
                "segments": t["segments"],
                "sentiment_results": [{"label": "neutral", "score": 0.5} for _ in t["segments"]],
                "intent_results": [],
                "results": {},
                "llm": {},
                "risks": {},
            }

        # Enrich for dashboard convenience (serializable)
        rdict["call_id"] = t.get("id", "UNKNOWN")
        rdict["title"] = t.get("title", rdict["call_id"])
        rdict["meta"] = t.get("meta", {})
        # Keep original transcript meta if present
        rdict.setdefault("demo_meta", {})["transcript_id"] = t["id"]
        reports.append(rdict)

    return tuple(reports)


def _generate_fallback_reports(transcripts: list[dict]) -> list[dict]:
    """Very minimal synthetic reports (only if pipeline import/run fails completely)."""
    reports = []
    for t in transcripts:
        r = {
            "call_id": t["id"],
            "title": t["title"],
            "meta": t.get("meta", {}),
            "segments": t["segments"],
            "sentiment_results": [{"label": "neutral", "score": 0.5, "score_pos": 0.3, "score_neg": 0.2} for _ in t["segments"]],
            "intent_results": [("information_request", 0.6)],
            "summary": {"text": "Fallback synthetic summary."},
            "topics": {},
            "insights": {},
            "risks": {"risk_level": "low"},
            "results": {
                "qa": {"overall_qa_score": 75, "passed": True, "risk_level": "low", "compliance_flags": []},
                "agent_performance": {"agent": {"empathy_score": 0.65, "compliance_flags": []}},
                "alerts": [],
            },
            "llm": {},
            "processing_time_s": 0.1,
        }
        reports.append(r)
    return reports


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS (pure, serializable, evidence-aware)
# ---------------------------------------------------------------------------

def extract_call_summary(report: dict[str, Any]) -> str:
    """Best-effort Swedish summary. Prefers LLM actionable_summary, then summary.text, then fallback."""
    llm = report.get("llm") or {}
    if isinstance(llm, dict):
        act = llm.get("actionable_summary") or {}
        if isinstance(act, dict) and act.get("problem"):
            return f"{act.get('problem', '')} | Kundens läge: {act.get('final_customer_state', '')}"
        if llm.get("trajectory", {}).get("summary"):
            return llm["trajectory"]["summary"]

    summ = report.get("summary") or {}
    if isinstance(summ, dict):
        if summ.get("text"):
            return str(summ["text"])
        if summ.get("summary"):
            return str(summ["summary"])

    # Fallback: first 1-2 segments
    segs = report.get("segments", [])
    if segs:
        return " | ".join(s.get("text", "")[:80] for s in segs[:2])
    return "Ingen sammanfattning tillgänglig."


def get_overall_sentiment(report: dict[str, Any]) -> dict[str, Any]:
    """Return {'label': str, 'score': float, 'source': str} for the call.

    Prefers majority vote on sentiment_results (local), falls back to llm trajectory slope or risks.
    Always returns a dict so UI can be consistent.
    """
    sents = report.get("sentiment_results") or []
    if sents:
        labels = [str(s.get("label", "neutral")).lower() for s in sents if isinstance(s, dict)]
        if labels:
            from collections import Counter

            majority = Counter(labels).most_common(1)[0][0]
            # crude numeric
            score_map = {"positiv": 0.8, "positive": 0.8, "neutral": 0.0, "negativ": -0.7, "negative": -0.7}
            avg_score = sum(score_map.get(l, 0.0) for l in labels) / len(labels)
            return {"label": majority, "score": round(avg_score, 3), "source": "local_sentiment_results"}

    # LLM trajectory customer slope
    llm = report.get("llm") or {}
    traj = llm.get("trajectory") or {}
    slope = traj.get("customer_sentiment_slope")
    if slope is not None:
        label = "positiv" if slope > 0.1 else ("negativ" if slope < -0.1 else "neutral")
        return {"label": label, "score": round(float(slope), 3), "source": "llm_trajectory"}

    # Risk-based fallback
    risks = report.get("risks") or {}
    rl = str(risks.get("risk_level", "")).lower()
    if "high" in rl or "critical" in rl:
        return {"label": "negativ", "score": -0.6, "source": "risks"}
    return {"label": "neutral", "score": 0.0, "source": "fallback"}


def _get_sentiment_score(s: dict) -> float:
    """Normalize various sentiment result shapes to [-1,1] float."""
    if not isinstance(s, dict):
        return 0.0
    if "score" in s and isinstance(s["score"], (int, float)):
        sc = float(s["score"])
        # if already -1..1 keep, if 0-1 assume positive lean or use label
        if -1.0 <= sc <= 1.0:
            return sc
        return (sc - 0.5) * 2 if 0 <= sc <= 1 else 0.0
    label = str(s.get("label", "")).lower()
    if "pos" in label:
        return 0.75
    if "neg" in label:
        return -0.75
    return 0.0


def enrich_segments_with_sentiment(
    segments: list[dict[str, Any]],
    sentiment_results: list[dict[str, Any]],
    *,
    emotion_results: list[dict[str, Any]] | None = None,
    compliance_risk: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return list of enriched segment dicts for UI rendering (timeline + transcript).

    Adds: turn_idx, sentiment_label, sentiment_score (-1..1), emotion scores, compliance flags.
    Safe if lengths mismatch (pads with neutral).
    """
    flagged_indices: set[int] = set()
    if isinstance(compliance_risk, dict):
        for item in compliance_risk.get("flagged_segments") or []:
            if isinstance(item, dict) and "segment_index" in item:
                flagged_indices.add(int(item["segment_index"]))

    enriched: list[dict] = []
    n = len(segments)
    sres = sentiment_results or []
    eres = emotion_results or []
    for i in range(n):
        seg = dict(segments[i])  # shallow copy
        sent = sres[i] if i < len(sres) else {}
        seg["turn_idx"] = i
        seg["sentiment_label"] = str(sent.get("label", "neutral")).lower()
        seg["sentiment_score"] = _get_sentiment_score(sent)
        # Evidence hints (for highlighting nyckelmoment)
        seg["is_negative_peak"] = seg["sentiment_score"] < -0.5
        seg["has_compliance_flag"] = i in flagged_indices

        emo = eres[i] if i < len(eres) else {}
        if isinstance(emo, dict) and emo.get("scores"):
            seg["emotion"] = dict(emo["scores"])
        elif isinstance(emo, dict) and emo.get("primary"):
            seg["emotion"] = {str(emo["primary"]): 0.75}

        enriched.append(seg)
    return enriched


def compute_kpis(reports: list[dict[str, Any]], filters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute 6-8 high-level KPIs from (filtered) reports. Used for clickable cards.

    Returns dict with totals, ratios, counts, hot-ish aggregates. All values serializable.
    """
    f = filters or {}
    filtered = filter_reports(reports, f) if f else reports
    n = len(filtered) if filtered else 1

    total = len(filtered)
    sentiments = [get_overall_sentiment(r) for r in filtered]
    pos = sum(1 for s in sentiments if "pos" in s["label"])
    neg = sum(1 for s in sentiments if "neg" in s["label"])
    pos_pct = round(pos / max(1, total) * 100)
    neg_pct = round(neg / max(1, total) * 100)

    # QA avg (Fas4)
    qa_scores = []
    for r in filtered:
        qa = (r.get("results") or {}).get("qa") or (r.get("results") or {}).get("compliance_qa") or {}
        sc = qa.get("overall_qa_score")
        if isinstance(sc, (int, float)):
            qa_scores.append(float(sc))
    qa_avg = round(sum(qa_scores) / len(qa_scores), 1) if qa_scores else None

    # Alerts count
    alert_count = 0
    for r in filtered:
        al = (r.get("results") or {}).get("alerts") or []
        alert_count += len(al) if isinstance(al, list) else 0

    # Simple hot topics count (from results or llm)
    hot_count = 0
    for r in filtered:
        topics = r.get("topics") or {}
        if topics:
            hot_count += len(topics.get("topics", [])) if isinstance(topics.get("topics"), list) else 1
        llm = r.get("llm") or {}
        if llm.get("refined_aspects"):
            hot_count += len(llm["refined_aspects"])

    # Risky calls
    risky = sum(
        1
        for r in filtered
        if str(((r.get("results") or {}).get("qa") or {}).get("risk_level", "")).lower() in {"high", "critical"}
        or str((r.get("risks") or {}).get("risk_level", "")).lower() in {"high", "critical"}
    )

    return {
        "total_calls": total,
        "pos_pct": pos_pct,
        "neg_pct": neg_pct,
        "qa_avg": qa_avg,
        "alerts_count": alert_count,
        "hot_topics_count": hot_count,
        "risky_calls": risky,
        "avg_processing_s": round(sum(r.get("processing_time_s", 0) for r in filtered) / max(1, total), 2),
    }


def filter_reports(reports: list[dict[str, Any]], filters: dict[str, Any]) -> list[dict[str, Any]]:
    """Apply persistent filters (from session_state). Supports:
    sentiment_filter: 'positiv' | 'negativ' | 'all'
    agent_filter: str or None
    has_qa_fail: bool
    min_risk: 'low'|'medium'|'high' etc (maps to qa/risks)
    topic_filter: str (simple contains in topics or aspects)
    search: str (in title/id/summary)
    """
    if not filters:
        return list(reports)

    out = []
    sf = filters.get("sentiment_filter", "all")
    af = filters.get("agent_filter")
    has_qa_fail = filters.get("has_qa_fail")
    min_risk = filters.get("min_risk")
    topic_f = (filters.get("topic_filter") or "").lower()
    search = (filters.get("search") or "").lower()

    for r in reports:
        sent = get_overall_sentiment(r)
        if sf != "all" and sf not in sent["label"]:
            continue

        agent = (r.get("meta") or {}).get("agent") or ""
        if af and af.lower() not in agent.lower():
            continue

        qa = (r.get("results") or {}).get("qa") or (r.get("results") or {}).get("compliance_qa") or {}
        if has_qa_fail is True and qa.get("passed") is True:
            continue
        if has_qa_fail is False and qa.get("passed") is False:
            continue

        if min_risk:
            qar = str(qa.get("risk_level", "")).lower()
            rr = str((r.get("risks") or {}).get("risk_level", "")).lower()
            if min_risk == "high" and not ("high" in qar or "critical" in qar or "high" in rr):
                continue
            if min_risk == "medium" and not (qar or rr):
                continue

        if topic_f:
            text_blob = " ".join(
                [
                    str((r.get("topics") or {}).get("topics", [])),
                    str((r.get("llm") or {}).get("refined_aspects", [])),
                    r.get("title", ""),
                ]
            ).lower()
            if topic_f not in text_blob:
                continue

        if search:
            blob = " ".join(
                [
                    r.get("call_id", ""),
                    r.get("title", ""),
                    extract_call_summary(r),
                    " ".join(s.get("text", "") for s in r.get("segments", [])[:3]),
                ]
            ).lower()
            if search not in blob:
                continue

        out.append(r)
    return out


def get_agent_leaderboard(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Simple leaderboard aggregated from agent_performance + llm agent_assessment."""
    from collections import defaultdict

    by_agent: dict[str, dict] = defaultdict(lambda: {"calls": 0, "empathy_sum": 0.0, "qa_sum": 0.0, "qa_n": 0, "coaching": 0})
    for r in reports:
        agent = (r.get("meta") or {}).get("agent") or "Okänd"
        by_agent[agent]["calls"] += 1

        ap = (r.get("results") or {}).get("agent_performance") or {}
        a = ap.get("agent") if isinstance(ap, dict) else {}
        if isinstance(a, dict) and a.get("empathy_score") is not None:
            by_agent[agent]["empathy_sum"] += float(a["empathy_score"])

        assess = (r.get("results") or {}).get("agent_assessment") or (r.get("llm") or {}).get("agent_assessment") or {}
        if isinstance(assess, dict) and assess.get("empathy_score") is not None:
            by_agent[agent]["empathy_sum"] += float(assess["empathy_score"])  # double weight if both

        qa = (r.get("results") or {}).get("qa") or {}
        if isinstance(qa, dict) and qa.get("overall_qa_score") is not None:
            by_agent[agent]["qa_sum"] += float(qa["overall_qa_score"])
            by_agent[agent]["qa_n"] += 1

        # crude coaching count
        recs = (assess or {}).get("specific_coaching_recommendations") or []
        by_agent[agent]["coaching"] += len(recs) if isinstance(recs, list) else 0

    board = []
    for name, agg in by_agent.items():
        n = agg["calls"]
        board.append(
            {
                "agent": name,
                "calls": n,
                "avg_empathy": round(agg["empathy_sum"] / max(1, n * 1.5), 2),  # rough
                "avg_qa": round(agg["qa_sum"] / max(1, agg["qa_n"]), 1) if agg["qa_n"] else None,
                "coaching_recs": agg["coaching"],
            }
        )
    board.sort(key=lambda x: (x["avg_empathy"] or 0, x["avg_qa"] or 0), reverse=True)
    return board


def collect_all_alerts(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten all alerts across reports for the Alerts Panel (with call context)."""
    all_a = []
    for r in reports:
        alerts = (r.get("results") or {}).get("alerts") or []
        if isinstance(alerts, list):
            for a in alerts:
                if isinstance(a, dict):
                    aa = dict(a)
                    aa["call_id"] = r.get("call_id")
                    aa["title"] = r.get("title")
                    all_a.append(aa)
    return all_a


def get_hot_topics(reports: list[dict[str, Any]], top_k: int = 8) -> list[dict[str, Any]]:
    """Lightweight hot topics from per-call + aggregator style (no full aggregate call for speed in MVP)."""
    from collections import Counter

    topic_counts: Counter = Counter()
    for r in reports:
        # from topics
        tops = r.get("topics") or {}
        if isinstance(tops, dict):
            for t in tops.get("topics", []) or []:
                if isinstance(t, dict):
                    topic_counts[t.get("topic", str(t))] += 1
                else:
                    topic_counts[str(t)] += 1
        # from llm aspects
        aspects = (r.get("llm") or {}).get("refined_aspects") or []
        for asp in aspects:
            if isinstance(asp, dict):
                topic_counts[asp.get("aspect", "okänt")] += 1
    return [{"topic": t, "volume": c} for t, c in topic_counts.most_common(top_k)]


# ---------------------------------------------------------------------------
# UPLOAD / INGEST HELPERS
# ---------------------------------------------------------------------------

def ingest_uploaded_report(uploaded: Any) -> list[dict]:
    """Handle st.file_uploader result (bytes or str) that is a full report JSON or list of reports.

    Supports:
    - Single report dict (from report.to_dict())
    - List of reports
    - Legacy {"calls": [...]} (wrapped)
    Returns normalized list[dict] ready for session.
    """
    if uploaded is None:
        return []
    if isinstance(uploaded, (bytes, bytearray)):
        try:
            data = json.loads(uploaded.decode("utf-8"))
        except Exception:
            return []
    elif isinstance(uploaded, str):
        try:
            data = json.loads(uploaded)
        except Exception:
            return []
    else:
        data = uploaded  # already parsed?

    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    if isinstance(data, dict):
        if "calls" in data and isinstance(data["calls"], list):
            return data["calls"]
        if "segments" in data or "sentiment_results" in data:  # looks like single report
            return [data]
        # wrapped?
        if "report" in data:
            return [data["report"]] if isinstance(data["report"], dict) else []
    return []


# ---------------------------------------------------------------------------
# SMALL UTILS FOR UI
# ---------------------------------------------------------------------------

def make_serializable(obj: Any) -> Any:
    """Best effort to make objects (incl. Pydantic) JSON-safe for st.session_state / download."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, (list, tuple)):
        return [make_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    return obj


def get_evidence_quotes(report: dict, max_quotes: int = 5) -> list[str]:
    """Collect concrete evidence quotes/spans from llm, qa, alerts, agent_assessment for beautiful display."""
    quotes: list[str] = []
    llm = report.get("llm") or {}

    def add_ev(ev):
        if isinstance(ev, dict) and ev.get("text"):
            quotes.append(str(ev["text"])[:120])
        elif isinstance(ev, str):
            quotes.append(ev[:120])

    # LLM evidence
    for key in ("actionable_summary", "root_cause", "agent_assessment"):
        item = llm.get(key) or {}
        if isinstance(item, dict):
            for ev in item.get("evidence_spans", []) or []:
                add_ev(ev)
            for rec in item.get("specific_coaching_recommendations", []) or []:
                if isinstance(rec, dict):
                    for ev in rec.get("evidence_spans", []) or []:
                        add_ev(ev)

    # QA
    qa = (report.get("results") or {}).get("qa") or {}
    for cr in (qa.get("criteria_results") or []):
        if isinstance(cr, dict):
            for ev in cr.get("evidence", []) or []:
                add_ev(ev)

    # Alerts
    for al in (report.get("results") or {}).get("alerts") or []:
        if isinstance(al, dict):
            for ev in al.get("evidence_spans", []) or []:
                add_ev(ev)

    return list(dict.fromkeys(quotes))[:max_quotes]  # dedup preserve order


# Note: Future React migration comment (per plan):
# These functions map 1:1 to backend responses. In React they live in lib/reportUtils.ts
# e.g. export const computeKPIs = (reports: Report[], filters?: Filters) => ...
# Components will receive props={report} and use <Timeline segments={enriched} onSegmentClick=... />
