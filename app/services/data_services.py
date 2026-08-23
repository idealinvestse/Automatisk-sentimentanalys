"""Canned demo transcripts and small helpers for unit tests.

Primary UI is Next.js `webui/`, which ports `DEMO_TRANSCRIPTS` to
`webui/src/lib/demo-transcripts.ts`. These Python helpers stay as the
source of that contract.

Usage:
    from app.services.data_services import (
        get_demo_transcripts, get_overall_sentiment, filter_reports,
    )
"""

from __future__ import annotations

from typing import Any

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
            {
                "start": 0.0,
                "end": 8.0,
                "text": "Hej, jag heter Anna på kundtjänst, hur kan jag hjälpa dig idag?",
                "speaker": "Agent",
            },
            {
                "start": 8.0,
                "end": 18.0,
                "text": "Hej Anna, jag har fått en faktura på 890 kr som jag inte förstår. Det står att jag har ringt internationellt men det har jag inte.",
                "speaker": "Kund",
            },
            {
                "start": 18.0,
                "end": 32.0,
                "text": "Tack för att du ringer in. Jag förstår att det känns frustrerande. Kan jag få ditt kundnummer eller personnummer så kollar jag upp det direkt?",
                "speaker": "Agent",
            },
            {
                "start": 32.0,
                "end": 42.0,
                "text": "Ja, det är 19851203-1234. Och jag har aldrig ringt utomlands, jag lovar.",
                "speaker": "Kund",
            },
            {
                "start": 42.0,
                "end": 65.0,
                "text": "Tack, jag ser nu i systemet att det finns en debitering från ett samtal till +49 den 12 maj. Men jag ser också att du har ett abonnemang som inkluderar EU-samtal. Det verkar som en felkodning i faktureringssystemet. Jag krediterar 890 kr nu direkt och skickar en rättad faktura.",
                "speaker": "Agent",
            },
            {
                "start": 65.0,
                "end": 75.0,
                "text": "Oj, tack! Det var snabbt. Hur lång tid tar det innan det syns på kontot?",
                "speaker": "Kund",
            },
            {
                "start": 75.0,
                "end": 88.0,
                "text": "Det syns på nästa faktura eller som kredit inom 3-5 vardagar. Jag lägger också en notering så att det inte händer igen. Är det något mer jag kan hjälpa dig med idag?",
                "speaker": "Agent",
            },
            {
                "start": 88.0,
                "end": 95.0,
                "text": "Nej, det var allt. Tack för hjälpen, Anna – du var jättebra!",
                "speaker": "Kund",
            },
            {"start": 95.0, "end": 102.0, "text": "Tack själv, ha en bra dag!", "speaker": "Agent"},
        ],
    },
    {
        "id": "CALL-002",
        "title": "Lång väntetid + arg kund – eskaleringsrisk",
        "meta": {"agent": "Agent-Bengt", "duration_s": 310, "category": "complaint"},
        "segments": [
            {
                "start": 0.0,
                "end": 5.0,
                "text": "Ja hallå? Jag har väntat i 45 minuter i kön!",
                "speaker": "Kund",
            },
            {
                "start": 5.0,
                "end": 12.0,
                "text": "Hej, tack för att du väntar. Mitt namn är Bengt. Vad gäller ditt ärende?",
                "speaker": "Agent",
            },
            {
                "start": 12.0,
                "end": 25.0,
                "text": "Jag ringde för att säga upp mitt abonnemang för två veckor sedan och jag har fortfarande inte fått bekräftelse. Och nu kommer en ny faktura ändå! Detta är skandal!",
                "speaker": "Kund",
            },
            {
                "start": 25.0,
                "end": 35.0,
                "text": "Okej, jag förstår att du är upprörd. Men jag behöver ditt kundnummer för att kunna titta.",
                "speaker": "Agent",
            },
            {
                "start": 35.0,
                "end": 45.0,
                "text": "Jag har redan gett det i kön! Varför kan ni inte ha koll? Jag vill tala med en chef nu!",
                "speaker": "Kund",
            },
            {
                "start": 45.0,
                "end": 58.0,
                "text": "Jag kan inte koppla dig till chef direkt. Låt mig först kolla status på uppsägningen. Kan du upprepa kundnumret?",
                "speaker": "Agent",
            },
            {
                "start": 58.0,
                "end": 68.0,
                "text": "19851203-1234. Och jag vill ha skriftlig bekräftelse inom 24 timmar annars kontaktar jag Konsumentverket och ARN!",
                "speaker": "Kund",
            },
            {
                "start": 68.0,
                "end": 82.0,
                "text": "Okej, jag ser att uppsägningen registrerades den 14 maj men bekräftelsen gick inte iväg pga tekniskt fel. Jag skickar den nu manuellt och krediterar fakturan. Men jag kan tyvärr inte göra mer idag.",
                "speaker": "Agent",
            },
            {
                "start": 82.0,
                "end": 90.0,
                "text": "Det här duger inte. Jag är så less på er. Ni hör av er.",
                "speaker": "Kund",
            },
            {"start": 90.0, "end": 95.0, "text": "Tack för samtalet.", "speaker": "Agent"},
        ],
    },
    {
        "id": "CALL-003",
        "title": "Tekniskt fel + compliance near-miss (QA-flagg)",
        "meta": {"agent": "Agent-Cecilia", "duration_s": 480, "category": "tech_support"},
        "segments": [
            {
                "start": 0.0,
                "end": 4.0,
                "text": "Tjenare, det är Cecilia på support.",
                "speaker": "Agent",
            },
            {
                "start": 4.0,
                "end": 15.0,
                "text": "Hej, min router har varit nere hela helgen. Jag kan inte jobba. Jag har ringt tidigare och fick löfte om att någon skulle komma ut men ingenting har hänt.",
                "speaker": "Kund",
            },
            {
                "start": 15.0,
                "end": 22.0,
                "text": "Okej, tråkigt att höra. Har du provat att starta om routern?",
                "speaker": "Agent",
            },
            {
                "start": 22.0,
                "end": 30.0,
                "text": "Ja, tre gånger! Och jag har bytt sladd. Det är ert fel, inte mitt.",
                "speaker": "Kund",
            },
            {
                "start": 30.0,
                "end": 45.0,
                "text": "Förstår. Jag kollar i systemet – din linje visar röd sedan fredag. Jag bokar en tekniker till imorgon mellan 8-12. Bekräftar du adressen Storgatan 12?",
                "speaker": "Agent",
            },
            {
                "start": 45.0,
                "end": 52.0,
                "text": "Ja, det stämmer. Men jag vill ha kompensation för stilleståndet. Jag har förlorat jobbintäkter.",
                "speaker": "Kund",
            },
            {
                "start": 52.0,
                "end": 68.0,
                "text": "Vi har tyvärr ingen policy för det just nu. Men jag kan ge dig 50 kr rabatt på nästa faktura. Är det okej?",
                "speaker": "Agent",
            },
            {
                "start": 68.0,
                "end": 78.0,
                "text": "50 kr? Det är ju ingenting. Ni har förstört min helg. Jag vill ha minst 300 kr eller så lämnar jag er.",
                "speaker": "Kund",
            },
            {
                "start": 78.0,
                "end": 92.0,
                "text": "Låt mig se vad jag kan göra... Okej, jag lägger in 200 kr goodwill-kredit manuellt. Och tekniker imorgon. Tack för tålamodet.",
                "speaker": "Agent",
            },
            {
                "start": 92.0,
                "end": 100.0,
                "text": "Okej, det får duga. Men se till att det blir rätt denna gången.",
                "speaker": "Kund",
            },
            {"start": 100.0, "end": 108.0, "text": "Absolut. Ha en bra dag.", "speaker": "Agent"},
        ],
    },
    {
        "id": "CALL-004",
        "title": "Betalningsproblem + root cause (LLM-berikad)",
        "meta": {"agent": "Agent-Daniel", "duration_s": 390, "category": "billing"},
        "segments": [
            {
                "start": 0.0,
                "end": 6.0,
                "text": "Hej, Daniel här. Vad kan jag stå till tjänst med?",
                "speaker": "Agent",
            },
            {
                "start": 6.0,
                "end": 18.0,
                "text": "Jag har fått påminnelse om obetald faktura men jag betalade den förra månaden. Varför kommer det här?",
                "speaker": "Kund",
            },
            {
                "start": 18.0,
                "end": 30.0,
                "text": "Låt mig kolla. Jag ser att betalningen från 3 april inte har matchats mot rätt faktura i systemet. Det är ett känt problem just nu med vår bankkoppling.",
                "speaker": "Agent",
            },
            {
                "start": 30.0,
                "end": 40.0,
                "text": "Men jag har kvitto! Jag kan inte ha det här hängande över mig. Det påverkar min kreditvärdighet.",
                "speaker": "Kund",
            },
            {
                "start": 40.0,
                "end": 55.0,
                "text": "Jag beklagar verkligen. Jag markerar fakturan som betald manuellt nu och lägger en spärr så att inga fler påminnelser går ut. Jag skickar också bekräftelse till din e-post.",
                "speaker": "Agent",
            },
            {
                "start": 55.0,
                "end": 65.0,
                "text": "Okej... Men hur kunde det bli så här? Har ni inte koll på era system?",
                "speaker": "Kund",
            },
            {
                "start": 65.0,
                "end": 78.0,
                "text": "Det är ett internt IT-problem som vår leverantör håller på att fixa. Vi har haft flera fall den här veckan. Jag lägger en intern incidentrapport så att det inte drabbar fler.",
                "speaker": "Agent",
            },
            {
                "start": 78.0,
                "end": 88.0,
                "text": "Tack. Jag hoppas det löser sig fort. Annars byter jag operatör.",
                "speaker": "Kund",
            },
            {
                "start": 88.0,
                "end": 95.0,
                "text": "Förstår. Är det något annat jag kan hjälpa till med medan vi har kontakt?",
                "speaker": "Agent",
            },
        ],
    },
    {
        "id": "CALL-005",
        "title": "De-eskalering + lätt upsell (positiv vändning)",
        "meta": {"agent": "Agent-Erika", "duration_s": 275, "category": "retention"},
        "segments": [
            {
                "start": 0.0,
                "end": 7.0,
                "text": "Hej, det är Erika. Jag såg att du ringde angående ditt abonnemang.",
                "speaker": "Agent",
            },
            {
                "start": 7.0,
                "end": 18.0,
                "text": "Ja, jag funderar på att säga upp. Priset har gått upp och jag använder det knappt längre.",
                "speaker": "Kund",
            },
            {
                "start": 18.0,
                "end": 28.0,
                "text": "Jag förstår. Många upplever samma sak just nu. Får jag fråga vad du använder mest – mobil eller bredband?",
                "speaker": "Agent",
            },
            {
                "start": 28.0,
                "end": 35.0,
                "text": "Främst mobilen. Bredbandet har jag via jobbet.",
                "speaker": "Kund",
            },
            {
                "start": 35.0,
                "end": 48.0,
                "text": "Perfekt. Vi har just nu ett erbjudande där du kan behålla mobilt bredband + 100 GB för 199 kr/mån i 6 månader om du behåller abonnemanget. Det är 30 % lägre än nuvarande pris.",
                "speaker": "Agent",
            },
            {
                "start": 48.0,
                "end": 58.0,
                "text": "Hmm, 199 låter bättre. Men jag vill inte bindas i 24 månader igen.",
                "speaker": "Kund",
            },
            {
                "start": 58.0,
                "end": 70.0,
                "text": "Ingen bindningstid på det här erbjudandet. Du kan säga upp när som helst efter 6 månader. Vill du att jag aktiverar det nu?",
                "speaker": "Agent",
            },
            {
                "start": 70.0,
                "end": 78.0,
                "text": "Okej, kör på. Men bara om det verkligen blir 199.",
                "speaker": "Kund",
            },
            {
                "start": 78.0,
                "end": 88.0,
                "text": "Klart det blir. Jag aktiverar det nu och skickar bekräftelse. Tack för att du stannar hos oss – uppskattas!",
                "speaker": "Agent",
            },
            {"start": 88.0, "end": 93.0, "text": "Tack själv. Hej då.", "speaker": "Kund"},
        ],
    },
]


def get_demo_transcripts() -> list[dict[str, Any]]:
    """Return a copy of the canned demo transcripts (safe to mutate by caller)."""
    return [t.copy() for t in DEMO_TRANSCRIPTS]


def _generate_fallback_reports(transcripts: list[dict]) -> list[dict]:
    """Very minimal synthetic reports (only if pipeline import/run fails completely)."""
    reports = []
    for t in transcripts:
        r = {
            "call_id": t["id"],
            "title": t["title"],
            "meta": t.get("meta", {}),
            "segments": t["segments"],
            "sentiment_results": [
                {"label": "neutral", "score": 0.5, "score_pos": 0.3, "score_neg": 0.2}
                for _ in t["segments"]
            ],
            "intent_results": [("information_request", 0.6)],
            "summary": {"text": "Fallback synthetic summary."},
            "topics": {},
            "insights": {},
            "risks": {"risk_level": "low"},
            "results": {
                "qa": {
                    "overall_qa_score": 75,
                    "passed": True,
                    "risk_level": "low",
                    "compliance_flags": [],
                },
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
            score_map = {
                "positiv": 0.8,
                "positive": 0.8,
                "neutral": 0.0,
                "negativ": -0.7,
                "negative": -0.7,
            }
            avg_score = sum(score_map.get(label, 0.0) for label in labels) / len(labels)
            return {
                "label": majority,
                "score": round(avg_score, 3),
                "source": "local_sentiment_results",
            }

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


def filter_reports(reports: list[dict[str, Any]], filters: dict[str, Any]) -> list[dict[str, Any]]:
    """Apply optional filters. Supports:
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

        qa = (
            (r.get("results") or {}).get("qa")
            or (r.get("results") or {}).get("compliance_qa")
            or {}
        )
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
