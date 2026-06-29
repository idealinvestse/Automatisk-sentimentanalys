"""Aggregate analytics helpers for the Analys & trender tab."""

from __future__ import annotations

import csv
import io
from datetime import datetime
from typing import Any

from app.nicegui_dashboard.services.chart_data import (
    extract_agent_trend_rows,
    extract_trajectory_points,
)
from app.services.data_services import compute_kpis, get_hot_topics, get_overall_sentiment

_EMOTION_SV = {
    "joy": "Glädje",
    "sadness": "Sorg",
    "anger": "Ilska",
    "fear": "Rädsla",
    "disgust": "Avsky",
    "surprise": "Överraskning",
    "neutral": "Neutral",
}


def emotion_label_sv(name: str) -> str:
    """Map English emotion keys to Swedish display labels."""
    key = str(name).lower().strip()
    return _EMOTION_SV.get(key, name.capitalize())


def list_agent_options(reports: list[dict[str, Any]]) -> list[str]:
    """Unique agents for analytics filter dropdown."""
    agents = sorted({(r.get("meta") or {}).get("agent", "Okänd") for r in reports})
    return ["Alla"] + agents


def filter_reports_by_agent(
    reports: list[dict[str, Any]],
    agent_filter: str | None,
) -> list[dict[str, Any]]:
    """Optional agent filter used only in analytics tab."""
    if not agent_filter or agent_filter == "Alla":
        return list(reports)
    return [r for r in reports if (r.get("meta") or {}).get("agent", "Okänd") == agent_filter]


def compute_call_snapshot(report: dict[str, Any] | None) -> dict[str, Any]:
    """Rich per-call context for the analytics detail card."""
    if not report:
        return {}

    meta = report.get("meta") or {}
    results = report.get("results") or {}
    qa = results.get("qa") or results.get("compliance_qa") or {}
    sentiment = get_overall_sentiment(report)
    points = extract_trajectory_points(report)
    scores = [p["y"] for p in points if isinstance(p.get("y"), (int, float))]

    alerts = results.get("alerts") or []
    alert_count = len(alerts) if isinstance(alerts, list) else 0

    duration_s = meta.get("duration_s")
    if duration_s is None and report.get("segments"):
        segs = report["segments"]
        if segs:
            duration_s = max(float(s.get("end", 0)) for s in segs)

    trend = "Stabil"
    if len(scores) >= 2:
        delta = scores[-1] - scores[0]
        if delta > 0.15:
            trend = "Förbättras"
        elif delta < -0.15:
            trend = "Försämras"

    negative_peaks = sum(1 for s in scores if s < -0.4)

    return {
        "call_id": report.get("call_id") or report.get("id", "?"),
        "title": report.get("title", ""),
        "agent": meta.get("agent", "Okänd"),
        "category": meta.get("category") or meta.get("topic") or "—",
        "duration_s": duration_s,
        "segment_count": len(report.get("segments") or []),
        "sentiment_label": sentiment.get("label", "neutral"),
        "sentiment_score": sentiment.get("score", 0.0),
        "qa_score": qa.get("overall_qa_score"),
        "qa_passed": qa.get("passed"),
        "risk_level": qa.get("risk_level") or (report.get("risks") or {}).get("risk_level", "—"),
        "alert_count": alert_count,
        "trajectory_min": round(min(scores), 2) if scores else None,
        "trajectory_max": round(max(scores), 2) if scores else None,
        "trajectory_trend": trend,
        "negative_peaks": negative_peaks,
    }


def summarize_emotions(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Average emotion scores for selected call, sorted descending."""
    if not report:
        return []

    totals: dict[str, list[float]] = {}
    for seg in report.get("segments") or []:
        emotions = seg.get("emotion") or {}
        if not isinstance(emotions, dict):
            continue
        for name, score in emotions.items():
            try:
                totals.setdefault(name, []).append(float(score))
            except (TypeError, ValueError):
                continue

    if not totals:
        return []

    ranked = [
        {
            "emotion": name,
            "label_sv": emotion_label_sv(name),
            "avg": round(sum(vals) / len(vals), 2),
            "peak": round(max(vals), 2),
            "segments": len(vals),
        }
        for name, vals in totals.items()
    ]
    ranked.sort(key=lambda x: x["avg"], reverse=True)
    return ranked


def build_calls_overview_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Table rows combining trend metrics and sentiment per call."""
    rows: list[dict[str, Any]] = []
    for trend in extract_agent_trend_rows(reports):
        report = next(
            (r for r in reports if str(r.get("call_id") or r.get("id")) == trend["call_id"]),
            None,
        )
        sent = get_overall_sentiment(report) if report else {"label": "—", "score": 0}
        snap = compute_call_snapshot(report) if report else {}
        rows.append(
            {
                "call_id": trend["call_id"],
                "title": trend.get("title", ""),
                "agent": trend.get("agent", "Okänd"),
                "sentiment": sent.get("label", "—"),
                "sentiment_score": sent.get("score", 0),
                "empathy": trend.get("empathy"),
                "qa": trend.get("qa"),
                "escalation": trend.get("escalation", 0),
                "segments": snap.get("segment_count", "—"),
                "trend": snap.get("trajectory_trend", "—"),
            }
        )
    return rows


def compute_portfolio_kpis(
    reports: list[dict[str, Any]], filters: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Extend compute_kpis with agent/topic context for analytics header."""
    base = compute_kpis(reports, filters)
    agents = {(r.get("meta") or {}).get("agent", "Okänd") for r in reports}
    topics = get_hot_topics(reports, top_k=3)
    base["unique_agents"] = len(agents)
    base["top_topic"] = topics[0]["topic"] if topics else "—"
    base["avg_empathy"] = _avg_empathy(reports)
    return base


def aggregate_agent_stats(reports: list[dict[str, Any]], agent_id: str) -> dict[str, Any]:
    """Aggregate metrics for one agent across their calls."""
    from app.nicegui_dashboard.services.fas4_data import reports_for_agent

    agent_reports = reports_for_agent(reports, agent_id)
    rows = extract_agent_trend_rows(agent_reports)
    empathy_vals = [r["empathy"] for r in rows if r.get("empathy") is not None]
    qa_vals = [r["qa"] for r in rows if r.get("qa") is not None]
    alerts = sum(r.get("escalation", 0) for r in rows)
    return {
        "call_count": len(agent_reports),
        "avg_empathy": round(sum(empathy_vals) / len(empathy_vals), 2) if empathy_vals else None,
        "avg_qa": round(sum(qa_vals) / len(qa_vals), 1) if qa_vals else None,
        "alert_count": alerts,
    }


def qa_problem_calls(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Calls with QA failure or elevated risk."""
    problems: list[dict[str, Any]] = []
    for report in reports:
        qa = (report.get("results") or {}).get("qa") or {}
        risk = str(qa.get("risk_level", "")).lower()
        passed = qa.get("passed")
        if passed is False or risk in ("high", "critical", "medium"):
            problems.append(
                {
                    "call_id": report.get("call_id") or report.get("id", "?"),
                    "title": report.get("title", ""),
                    "agent": (report.get("meta") or {}).get("agent", "Okänd"),
                    "qa_score": qa.get("overall_qa_score", "—"),
                    "risk_level": qa.get("risk_level", "—"),
                    "passed": "Nej" if passed is False else "Ja",
                }
            )
    return problems


def total_pii_events(reports: list[dict[str, Any]]) -> int:
    """Sum PII redaction events across all loaded reports."""
    total = 0
    for report in reports:
        pii = (report.get("results") or {}).get("pii_redaction") or {}
        total += int(pii.get("total_redacted", 0) or 0)
    return total


def _avg_empathy(reports: list[dict[str, Any]]) -> float | None:
    rows = extract_agent_trend_rows(reports)
    values = [r["empathy"] for r in rows if r.get("empathy") is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 2)


_OVERVIEW_CSV_COLUMNS: list[tuple[str, str]] = [
    ("call_id", "ID"),
    ("title", "Ämne"),
    ("agent", "Agent"),
    ("sentiment", "Sentiment"),
    ("sentiment_score", "Sentimentpoäng"),
    ("empathy", "Empati"),
    ("qa", "QA"),
    ("escalation", "Aviseringar"),
    ("segments", "Segment"),
    ("trend", "Trend"),
]


def overview_rows_to_csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    """Serialize analytics overview table rows to UTF-8 CSV (Excel-friendly BOM)."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow([label for _, label in _OVERVIEW_CSV_COLUMNS])
    for row in rows:
        writer.writerow([row.get(key, "") for key, _ in _OVERVIEW_CSV_COLUMNS])
    return buf.getvalue().encode("utf-8-sig")


def overview_csv_filename() -> str:
    """Timestamped default filename for overview export."""
    return f"samtalsöversikt_{datetime.now():%Y%m%d_%H%M%S}.csv"
