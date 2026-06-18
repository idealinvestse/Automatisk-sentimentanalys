"""Fas 4 data helpers for NiceGUI dashboard – local extraction + optional API.

Aggregates agent performance, QA, hot topics, alerts and semantic search from
loaded reports, with async API fallback when backend is connected.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.services.data_services import collect_all_alerts, get_agent_leaderboard, get_hot_topics

logger = logging.getLogger(__name__)


def reports_to_segments_list(reports: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Extract segment lists from reports for Fas 4 API payloads."""
    return [list(r.get("segments") or []) for r in reports if r.get("segments")]


def list_agent_ids(reports: list[dict[str, Any]]) -> list[str]:
    """Unique agent names from report meta, sorted."""
    agents = sorted(
        {
            str((r.get("meta") or {}).get("agent") or "Okänd")
            for r in reports
        }
    )
    return agents


def reports_for_agent(reports: list[dict[str, Any]], agent_id: str) -> list[dict[str, Any]]:
    """Filter reports belonging to one agent."""
    return [
        r
        for r in reports
        if str((r.get("meta") or {}).get("agent") or "Okänd") == agent_id
    ]


def local_agent_metrics(agent_id: str, reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate agent metrics from per-call report.results (demo/API fallback)."""
    agent_reports = reports_for_agent(reports, agent_id)
    if not agent_reports:
        return {
            "call_count": 0,
            "averages": {},
            "trend_empathy": "unknown",
            "compliance_flags": [],
        }

    empathy_vals: list[float] = []
    talk_vals: list[float] = []
    de_esc_vals: list[float] = []
    flags: list[str] = []

    for r in agent_reports:
        ap = (r.get("results") or {}).get("agent_performance") or {}
        agent_block = ap.get("agent") if isinstance(ap.get("agent"), dict) else ap
        if not isinstance(agent_block, dict):
            continue
        empathy = agent_block.get("empathy_score")
        if empathy is not None:
            empathy_vals.append(float(empathy))
        else:
            assess = (r.get("results") or {}).get("agent_assessment") or (
                (r.get("llm") or {}).get("agent_assessment")
            ) or {}
            if isinstance(assess, dict) and assess.get("empathy_score") is not None:
                empathy_vals.append(float(assess["empathy_score"]))
        if agent_block.get("talk_ratio") is not None:
            talk_vals.append(float(agent_block["talk_ratio"]))
        if agent_block.get("de_escalation_effectiveness") is not None:
            de_esc_vals.append(float(agent_block["de_escalation_effectiveness"]))
        for flag in agent_block.get("compliance_flags") or []:
            if flag and flag not in flags:
                flags.append(str(flag))

    def _avg(vals: list[float]) -> float | None:
        return round(sum(vals) / len(vals), 3) if vals else None

    trend = "stable"
    if len(empathy_vals) >= 2:
        first_half = sum(empathy_vals[: len(empathy_vals) // 2]) / max(1, len(empathy_vals) // 2)
        second_half = sum(empathy_vals[len(empathy_vals) // 2 :]) / max(
            1, len(empathy_vals) - len(empathy_vals) // 2
        )
        if second_half - first_half > 0.05:
            trend = "improving"
        elif first_half - second_half > 0.05:
            trend = "declining"

    return {
        "call_count": len(agent_reports),
        "averages": {
            "empathy_score": _avg(empathy_vals),
            "talk_ratio": _avg(talk_vals),
            "de_escalation_effectiveness": _avg(de_esc_vals),
        },
        "trend_empathy": trend,
        "compliance_flags": flags,
    }


def local_hot_topics_detailed(reports: list[dict[str, Any]], top_k: int = 10) -> list[dict[str, Any]]:
    """Hot topics with volume; enrich from insights aggregator fields when present."""
    topics: list[dict[str, Any]] = []
    seen: set[str] = set()

    for r in reports:
        agg = (r.get("results") or {}).get("insights") or r.get("insights") or {}
        if isinstance(agg, dict):
            for ht in agg.get("hot_topics") or []:
                if not isinstance(ht, dict):
                    continue
                topic = str(ht.get("topic", ""))
                if not topic or topic in seen:
                    continue
                seen.add(topic)
                topics.append(
                    {
                        "topic": topic,
                        "volume": ht.get("volume", 1),
                        "avg_sentiment": ht.get("avg_sentiment"),
                        "trend": ht.get("trend", "stable"),
                        "evidence_spans": ht.get("evidence_spans") or [],
                    }
                )

    if topics:
        topics.sort(key=lambda t: t.get("volume", 0), reverse=True)
        return topics[:top_k]

    basic = get_hot_topics(reports, top_k=top_k)
    return [
        {
            "topic": t["topic"],
            "volume": t.get("volume", 1),
            "avg_sentiment": None,
            "trend": "stable",
            "evidence_spans": [],
        }
        for t in basic
    ]


def local_qa_from_report(report: dict[str, Any] | None) -> dict[str, Any]:
    """Extract QA scorecard from report.results."""
    if not report:
        return {}
    qa = (report.get("results") or {}).get("qa") or (report.get("results") or {}).get(
        "compliance_qa"
    ) or {}
    return dict(qa) if isinstance(qa, dict) else {}


def local_alerts(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten alerts with call context."""
    return collect_all_alerts(reports)


def alert_dedup_key(alert: dict[str, Any]) -> str:
    """Stable key for dismiss/handled tracking."""
    parts = [
        str(alert.get("call_id", "")),
        str(alert.get("rule_id", "")),
        str(alert.get("message", ""))[:80],
    ]
    return "|".join(parts)


def active_alerts(
    reports: list[dict[str, Any]],
    dismissed_keys: list[str] | set[str],
) -> list[dict[str, Any]]:
    """Alerts excluding user-dismissed items."""
    dismissed = set(dismissed_keys)
    return [a for a in local_alerts(reports) if alert_dedup_key(a) not in dismissed]


def _tokenize_query(query: str) -> list[str]:
    return [t for t in re.split(r"\W+", query.lower()) if len(t) >= 3]


def local_semantic_search(
    query: str,
    reports: list[dict[str, Any]],
    *,
    top_k: int = 5,
    agent_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Keyword fallback when API semantic search is unavailable."""
    tokens = _tokenize_query(query)
    if not tokens:
        return []

    hits: list[dict[str, Any]] = []
    for idx, report in enumerate(reports):
        agent = str((report.get("meta") or {}).get("agent") or "")
        if agent_filter and agent != agent_filter:
            continue
        call_id = str(report.get("call_id") or report.get("id") or idx)
        segments = report.get("segments") or []
        score = 0.0
        highlights: list[str] = []
        evidence: list[dict[str, Any]] = []

        for seg in segments:
            text = str(seg.get("text", ""))
            lower = text.lower()
            matched = sum(1 for t in tokens if t in lower)
            if matched:
                score += matched / len(tokens)
                if len(highlights) < 3:
                    highlights.append(text[:160])
                if len(evidence) < 2:
                    evidence.append({"text": text[:200], "speaker": seg.get("speaker")})

        if score > 0:
            hits.append(
                {
                    "id": str(idx),
                    "call_id": call_id,
                    "score": round(score, 3),
                    "highlights": highlights,
                    "evidence_spans": evidence,
                    "metadata": {"agent": agent, "title": report.get("title", call_id)},
                }
            )

    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:top_k]


def resolve_call_id_from_hit(hit: dict[str, Any], reports: list[dict[str, Any]]) -> str | None:
    """Map semantic search hit to dashboard call_id."""
    if hit.get("call_id"):
        return str(hit["call_id"])
    raw_id = hit.get("id")
    if raw_id is not None:
        try:
            idx = int(raw_id)
            if 0 <= idx < len(reports):
                r = reports[idx]
                return str(r.get("call_id") or r.get("id") or "")
        except (TypeError, ValueError):
            pass
    return None


def format_evidence_spans(spans: list[Any], *, max_items: int = 3) -> str:
    """Short Swedish-friendly evidence summary for tables."""
    lines: list[str] = []
    for span in spans[:max_items]:
        if isinstance(span, dict) and span.get("text"):
            lines.append(f"«{str(span['text'])[:100]}»")
        elif isinstance(span, str):
            lines.append(f"«{span[:100]}»")
    return " · ".join(lines) if lines else "—"


def severity_color(severity: str) -> str:
    """Quasar color for alert severity chip."""
    s = str(severity).lower()
    if s in ("critical", "high"):
        return "negative"
    if s in ("medium", "warning"):
        return "warning"
    return "grey"


async def fetch_agent_performance(
    client: Any,
    agent_id: str,
    reports: list[dict[str, Any]],
    *,
    window: str = "7d",
) -> tuple[dict[str, Any], str]:
    """Try API; return (metrics, source)."""
    segments_list = reports_to_segments_list(reports_for_agent(reports, agent_id))
    if not segments_list:
        return local_agent_metrics(agent_id, reports), "local"
    try:
        resp = await client.get_agent_performance(agent_id, segments_list, window=window)
        metrics = resp.get("metrics")
        if not metrics:
            return local_agent_metrics(agent_id, reports), "local"
        return metrics, "api"
    except Exception as err:
        logger.debug("agent_performance API fallback: %s", err)
        return local_agent_metrics(agent_id, reports), "local"


async def fetch_hot_topics(
    client: Any,
    reports: list[dict[str, Any]],
    *,
    window: str = "7d",
) -> tuple[list[dict[str, Any]], str]:
    """Try API hot topics; fallback to local."""
    segments_list = reports_to_segments_list(reports)
    if not segments_list:
        return local_hot_topics_detailed(reports), "local"
    try:
        resp = await client.get_hot_topics(segments_list, window=window)
        topics = resp.get("hot_topics") or local_hot_topics_detailed(reports)
        return topics, "api"
    except Exception as err:
        logger.debug("hot_topics API fallback: %s", err)
        return local_hot_topics_detailed(reports), "local"


async def fetch_semantic_search(
    client: Any,
    query: str,
    reports: list[dict[str, Any]],
    *,
    top_k: int = 5,
    agent_filter: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Try API semantic search; fallback to keyword match."""
    segments_list = reports_to_segments_list(reports)
    if not query.strip():
        return [], "local"
    if not segments_list:
        return local_semantic_search(query, reports, top_k=top_k, agent_filter=agent_filter), "local"
    try:
        filters = {"agent": agent_filter} if agent_filter else None
        resp = await client.semantic_search(
            query,
            segments_list,
            top_k=top_k,
            filters=filters,
        )
        hits = resp.get("hits") or []
        for hit in hits:
            if isinstance(hit, dict) and "call_id" not in hit:
                cid = resolve_call_id_from_hit(hit, reports)
                if cid:
                    hit["call_id"] = cid
        return hits, "api"
    except Exception as err:
        logger.debug("semantic_search API fallback: %s", err)
        return local_semantic_search(query, reports, top_k=top_k, agent_filter=agent_filter), "local"


def agent_leaderboard_rows(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Leaderboard rows for agent performance tab."""
    return get_agent_leaderboard(reports)