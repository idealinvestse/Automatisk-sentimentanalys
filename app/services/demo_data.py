"""Demo data service for Streamlit MVP dashboard.

Provides realistic Swedish call center conversation examples (canned transcripts)
and a cached generator that runs the real CallAnalysisPipeline to produce
full CallAnalysisReport dicts. This ensures the UI always displays *evidence-based*
output from the actual backend (Fas 1-4 local engines + optional LLM).

Why this approach (per harmonized plan):
- Delivers "snabbt värde" immediately: dashboard shows real agent_performance,
  qa/compliance, alerts, local assessments without needing API server or LLM key.
- st.cache_data ensures we only pay the (one-time) model load + inference cost.
- Data shapes are identical to what /analyze_pipeline and analyze_segments return.
- Easy to extend with more scenarios or load from files later.
- Prepares for React migration: the returned dicts are the contract.

Usage:
    from app.services.demo_data import get_demo_reports, DEMO_CALLS
    reports = get_demo_reports(use_llm=False)  # list[dict] = report.to_dict()
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import streamlit as st

# Ensure project root on path when run via streamlit
import sys
from pathlib import Path
if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Realistic canned call center conversations (Swedish, spoken style)
# Each is a list of segment dicts with optional speaker for diarization-like role.
# Chosen to exercise different paths: positive, negative peaks, compliance miss,
# good de-escalation, root cause-ish (price + process), etc.
# ---------------------------------------------------------------------------

DEMO_CALLS: list[dict[str, Any]] = [
    {
        "id": "CALL-001",
        "title": "Fakturafråga – snabb lösning (positiv)",
        "timestamp": "2026-06-02T09:15:00Z",
        "duration_s": 185,
        "agent": "Anna",
        "segments": [
            {"text": "Hej, jag undrar över min senaste faktura, den verkar för hög.", "speaker": "kund"},
            {"text": "Hej! Vad roligt att du ringer. Jag heter Anna på kundtjänst. Jag förstår att det kan kännas jobbigt med oväntade kostnader. Kan jag få ditt kundnummer så kollar jag direkt?", "speaker": "agent"},
            {"text": "Ja, det är 123456.", "speaker": "kund"},
            {"text": "Tack. Jag ser att du haft ett extra dataabonnemang som aktiverades förra månaden. Det förklarar differensen. Vill du att jag tar bort det nu?", "speaker": "agent"},
            {"text": "Ja tack, det var det jag misstänkte. Kan du skicka bekräftelse på mail?", "speaker": "kund"},
            {"text": "Absolut, jag ordnar det med en gång. Du ska ha ett mail inom fem minuter. Är det något annat jag kan hjälpa dig med idag?", "speaker": "agent"},
            {"text": "Nej, det var allt. Tack för hjälpen, du var snabb och trevlig!", "speaker": "kund"},
            {"text": "Tack själv! Ha en bra dag.", "speaker": "agent"},
        ],
    },
    {
        "id": "CALL-002",
        "title": "Teknisk support – långdraget, låg empati (negativ)",
        "timestamp": "2026-06-02T10:42:00Z",
        "duration_s": 620,
        "agent": "Erik",
        "segments": [
            {"text": "Jag har ringt tre gånger nu, ingenting funkar. Min router är död.", "speaker": "kund"},
            {"text": "Okej, har du testat att starta om den?", "speaker": "agent"},
            {"text": "Ja, jag sa ju det förra gången. Det är inte det.", "speaker": "kund"},
            {"text": "Då får vi göra en felsökning. Kan du logga in på...", "speaker": "agent"},
            {"text": "Jag är så jävla trött på det här. Ni lovade att det skulle vara fixat igår!", "speaker": "kund"},
            {"text": "Jag förstår att du är frustrerad men vi måste följa protokollet.", "speaker": "agent"},
            {"text": "Protokoll? Jag vill ha en ny router nu, inte prata mer!", "speaker": "kund"},
            {"text": "Okej, då bokar jag en tekniker. Det blir om tre dagar.", "speaker": "agent"},
            {"text": "Tre dagar till? Det här är helt oacceptabelt.", "speaker": "kund"},
        ],
    },
    {
        "id": "CALL-003",
        "title": "Uppsägning + återvinning (blandad, bra de-eskalering)",
        "timestamp": "2026-06-02T14:05:00Z",
        "duration_s": 310,
        "agent": "Maria",
        "segments": [
            {"text": "Jag vill säga upp mitt abonnemang, det är för dyrt nu.", "speaker": "kund"},
            {"text": "Hej, jag heter Maria. Jag hör att du funderar på att lämna oss – det är tråkigt att höra. Får jag fråga vad som fått dig att tänka så?", "speaker": "agent"},
            {"text": "Priset har gått upp och jag får inte den hastighet jag betalar för.", "speaker": "kund"},
            {"text": "Jag förstår, prisökningar suger. Låt mig kolla ditt avtal och se om det finns ett bättre paket som passar dig bättre, så slipper vi säga upp.", "speaker": "agent"},
            {"text": "Okej... vad har ni?", "speaker": "kund"},
            {"text": "Du ligger på 300 Mbit nu. Vi har just nu ett lojaliteterbjudande på 500 Mbit för samma pris som du har idag plus en månads fri. Skulle det kännas bättre?", "speaker": "agent"},
            {"text": "Ja, det låter faktiskt rimligt. Tack för att du inte bara försökte behålla mig utan gav ett bra alternativ.", "speaker": "kund"},
            {"text": "Kul att höra! Jag fixar uppgraderingen nu direkt och skickar bekräftelse. Ring gärna om hastigheten inte förbättras.", "speaker": "agent"},
        ],
    },
    {
        "id": "CALL-004",
        "title": "Klagomål på bemötande + compliance miss (ingen hälsning)", "timestamp": "2026-06-02T11:20:00Z", "duration_s": 240, "agent": "Johan",
        "segments": [
            {"text": "Är det kundtjänst?", "speaker": "kund"},
            {"text": "Ja vad gäller det?", "speaker": "agent"},  # Intentional weak greeting for QA demo
            {"text": "Jag har blivit debiterad för två abonnemang fast jag bara har ett. Ni har tagit 800 kr extra.", "speaker": "kund"},
            {"text": "Det kan jag kolla. Vad är personnumret?", "speaker": "agent"},
            {"text": "Personnummer? Är det säkert?", "speaker": "kund"},
            {"text": "Vi måste ha det för att identifiera dig.", "speaker": "agent"},
            {"text": "Det här känns jättekonstigt. Jag vill ha en chef nu.", "speaker": "kund"},
            {"text": "Okej jag kan eskalera men först behöver jag...", "speaker": "agent"},
        ],
    },
    {
        "id": "CALL-005",
        "title": "Leveransförsening – bra empati + actionable root cause",
        "timestamp": "2026-06-02T16:55:00Z",
        "duration_s": 275,
        "agent": "Sara",
        "segments": [
            {"text": "Hej, mitt paket skulle ha kommit igår men det är inte här. Jag är jätteirriterad.", "speaker": "kund"},
            {"text": "Hej, jag heter Sara och jag förstår verkligen din frustration. Att vänta på något man behöver suger. Låt mig spåra det åt dig nu.", "speaker": "agent"},
            {"text": "Tack... det är en present till min mamma.", "speaker": "kund"},
            {"text": "Jag hör att det är viktigt. Systemet visar att försändelsen fastnat hos transportören i Stockholm. Jag bokar omleverans med express imorgon på vår bekostnad och ger dig 20% rabatt på nästa köp som goodwill.", "speaker": "agent"},
            {"text": "Wow, det var mer än jag hoppades på. Tack, du har räddat dagen.", "speaker": "kund"},
            {"text": "Det är jag glad för. Jag skickar tracking nu. Om det inte är framme imorgon ringer du mig direkt på det här numret.", "speaker": "agent"},
        ],
    },
]


def _hash_segments(segments: list[dict[str, Any]]) -> str:
    """Stable hash for cache key based on transcript content."""
    joined = "|".join(s.get("text", "") for s in segments)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


@st.cache_data(show_spinner="Kör pipeline på demo-samtal (endast första gången)...", ttl=3600)
def get_demo_reports(use_llm: bool = False, profile: str = "callcenter") -> list[dict[str, Any]]:
    """Generate (or retrieve cached) full CallAnalysisReport dicts for the demo calls.

    Runs the *real* CallAnalysisPipeline so that results contain:
    - sentiment_results aligned with segments
    - results["agent_performance"] (Fas 4.1)
    - results["qa"] / results["compliance_qa"] (Fas 4.2)
    - results["alerts"] (Fas 4.4.2)
    - results["agent_assessment"] (local rules + possibly llm)
    - llm[...] when use_llm=True and key available (falls back gracefully otherwise)

    The returned list has the exact same structure as pipeline.analyze_segments(...).to_dict()
    and what the API /analyze_pipeline returns (plus a little extra demo metadata).

    This is the single source of truth for the Streamlit MVP's "real data" mode.
    """
    from src.pipeline import CallAnalysisPipeline  # local import to avoid circular at module load

    reports: list[dict[str, Any]] = []
    for call in DEMO_CALLS:
        segs = call["segments"]
        cache_key = f"{call['id']}:{_hash_segments(segs)}:{use_llm}:{profile}"

        # We still wrap the per-call run; the outer @st.cache_data caches the whole list
        try:
            pipe = CallAnalysisPipeline(
                profile=profile,
                use_mistral_llm=bool(use_llm),
                deep_analysis=bool(use_llm),
            )
            report = pipe.analyze_segments(segs)
            report_dict = report.to_dict()
            # Enrich with demo metadata for UI convenience (does not affect real reports)
            report_dict["_demo_meta"] = {
                "id": call["id"],
                "title": call.get("title", call["id"]),
                "agent": call.get("agent", "Okänd"),
                "duration_s": call.get("duration_s", 0),
                "timestamp": call.get("timestamp"),
            }
            reports.append(report_dict)
            logger.info("Generated demo report for %s (llm=%s)", call["id"], use_llm)
        except Exception as e:
            logger.exception("Failed to generate demo report for %s: %s", call["id"], e)
            # Fallback minimal report so UI doesn't break
            reports.append(
                {
                    "segments": segs,
                    "sentiment_results": [{"label": "neutral", "score": 0.5} for _ in segs],
                    "intent_results": [],
                    "_demo_meta": {
                        "id": call["id"],
                        "title": call.get("title", call["id"]) + " (fallback)",
                        "agent": call.get("agent", "Okänd"),
                        "duration_s": call.get("duration_s", 0),
                    },
                    "results": {"error": str(e)},
                    "llm": {},
                    "error": True,
                }
            )
    return reports


def get_call_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Extract a compact call summary dict suitable for table / KPI computation.

    Works for both demo-enriched reports and raw pipeline reports.
    """
    meta = report.get("_demo_meta", {})
    call_id = meta.get("id") or report.get("id", "CALL-???")
    agent = meta.get("agent") or "Okänd"

    # Overall sentiment heuristic (majority or first)
    sent_list = report.get("sentiment_results", []) or []
    labels = [s.get("label", "neutral") for s in sent_list if isinstance(s, dict)]
    if labels:
        from collections import Counter
        overall = Counter(labels).most_common(1)[0][0]
    else:
        overall = "neutral"

    # QA
    qa = (report.get("results") or {}).get("qa") or (report.get("results") or {}).get("compliance_qa") or {}
    qa_score = qa.get("overall_qa_score") if isinstance(qa, dict) else None
    qa_passed = qa.get("passed") if isinstance(qa, dict) else None

    # Risk / alerts
    alerts = (report.get("results") or {}).get("alerts", []) or []
    risk = "low"
    if alerts:
        severities = [a.get("severity", "medium") for a in alerts if isinstance(a, dict)]
        if "critical" in severities:
            risk = "critical"
        elif "high" in severities:
            risk = "high"
        elif "medium" in severities:
            risk = "medium"

    # Agent empathy from local or llm
    ap = (report.get("results") or {}).get("agent_performance") or {}
    emp = 0.0
    if isinstance(ap, dict):
        emp = (ap.get("agent") or {}).get("empathy_score", 0.0) or 0.0
    assess = (report.get("results") or {}).get("agent_assessment") or {}
    if isinstance(assess, dict) and assess.get("empathy_score") is not None:
        emp = assess.get("empathy_score", emp)

    return {
        "id": call_id,
        "title": meta.get("title", call_id),
        "timestamp": meta.get("timestamp") or report.get("timestamp", ""),
        "duration_s": meta.get("duration_s", 0),
        "agent": agent,
        "overall_sentiment": overall,
        "qa_score": qa_score,
        "qa_passed": qa_passed,
        "risk_level": risk,
        "empathy_score": round(emp, 2) if emp else None,
        "num_alerts": len(alerts),
        "has_llm": bool((report.get("llm") or {}).get("meta", {}).get("llm_used") or report.get("llm", {}).get("actionable_summary")),
    }


def compute_dashboard_kpis(reports: list[dict[str, Any]], filters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute the 6-8 top KPIs for the main dashboard from (filtered) reports."""
    if filters is None:
        filters = {}
    # Very light filtering here; real filter logic lives in dashboard for reactivity
    filtered = reports  # caller should pre-filter for accuracy

    total = len(filtered)
    if total == 0:
        return {"total": 0}

    sentiments = []
    qa_scores = []
    alert_count = 0
    empathy_sum = 0.0
    empathy_n = 0
    for r in filtered:
        s = get_call_summary(r)
        sentiments.append(s["overall_sentiment"])
        if s["qa_score"] is not None:
            qa_scores.append(s["qa_score"])
        alert_count += s["num_alerts"]
        if s["empathy_score"] is not None:
            empathy_sum += s["empathy_score"]
            empathy_n += 1

    pos = sentiments.count("positiv") / total
    neg = sentiments.count("negativ") / total
    avg_qa = sum(qa_scores) / len(qa_scores) if qa_scores else None
    avg_emp = empathy_sum / empathy_n if empathy_n else None

    # Simple "hot topics" stub (real one would come from aggregator; here from intent/topics if present)
    hot = {}
    for r in filtered:
        for intent_item in r.get("intent_results", []) or []:
            if isinstance(intent_item, (list, tuple)) and intent_item:
                key = intent_item[0]
            elif isinstance(intent_item, dict):
                key = intent_item.get("intent") or intent_item.get("label")
            else:
                continue
            if key:
                hot[key] = hot.get(key, 0) + 1
    top_hot = sorted(hot.items(), key=lambda x: -x[1])[:3]

    return {
        "total_calls": total,
        "positive_pct": round(pos * 100),
        "negative_pct": round(neg * 100),
        "avg_qa_score": round(avg_qa, 1) if avg_qa is not None else None,
        "total_alerts": alert_count,
        "avg_empathy": round(avg_emp, 2) if avg_emp is not None else None,
        "top_hot_topics": top_hot,
        "escalated_or_high_risk": sum(1 for r in filtered if get_call_summary(r)["risk_level"] in ("high", "critical")),
    }


# Convenience for ad-hoc single call analysis (used by "Live-analys" and detail paste)
def analyze_single_segments(segments: list[dict[str, Any]], use_llm: bool = False, profile: str = "callcenter", llm_api_key: str | None = None) -> dict[str, Any]:
    """Run pipeline on arbitrary segments and return to_dict() + demo-like meta."""
    from src.pipeline import CallAnalysisPipeline

    pipe = CallAnalysisPipeline(profile=profile, use_mistral_llm=use_llm, deep_analysis=use_llm, llm_api_key=llm_api_key)
    report = pipe.analyze_segments(segments)
    d = report.to_dict()
    d["_demo_meta"] = {"id": "LIVE-" + hashlib.sha256(str(segments).encode()).hexdigest()[:8], "title": "Live-analys", "agent": "Live"}
    return d
