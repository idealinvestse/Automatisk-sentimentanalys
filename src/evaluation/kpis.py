"""Fas 4 evaluation KPIs (QA, coaching, topics, PII, alerts, cache)."""

from __future__ import annotations

from typing import Any


def compute_qa_score_consistency(qa_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Fas 4.2 KPI: share of QA results that pass with low/medium risk (stability proxy).

    When both ``rule_score`` and ``hybrid_score`` (or ``llm_score``) are present,
    also report pairwise agreement within 0.15 absolute score.
    """
    if not qa_results:
        return {"agreement": 0.0, "n": 0, "score_agreement": None}
    consistent = sum(
        1 for r in qa_results if r.get("passed") and r.get("risk_level") in ("low", "medium")
    )
    paired = 0
    agree = 0
    for r in qa_results:
        a = r.get("rule_score")
        b = r.get("hybrid_score", r.get("llm_score"))
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            paired += 1
            if abs(float(a) - float(b)) <= 0.15:
                agree += 1
    out: dict[str, Any] = {
        "agreement": round(consistent / len(qa_results), 3),
        "n": len(qa_results),
    }
    if paired:
        out["score_agreement"] = round(agree / paired, 3)
        out["n_paired_scores"] = paired
    else:
        out["score_agreement"] = None
    return out


def compute_coaching_precision(
    coaching_recs: list[dict[str, Any]], human_judged_good: list[bool] | None = None
) -> dict[str, Any]:
    """Fas 4.1.2 KPI: precision of coaching recommendations.

    With human labels: classic precision. Without: evidence-backed rate
    (recs that include ``evidence_spans`` or non-empty ``rationale``).
    """
    if not coaching_recs:
        return {"precision": 0.0, "n": 0, "note": "no recs"}
    if human_judged_good is not None and len(human_judged_good) == len(coaching_recs):
        good = sum(1 for g in human_judged_good if g)
        return {"precision": round(good / len(coaching_recs), 3), "n": len(coaching_recs)}
    with_ev = sum(
        1
        for r in coaching_recs
        if r.get("evidence_spans") or (isinstance(r.get("rationale"), str) and r["rationale"].strip())
    )
    return {
        "precision": round(with_ev / len(coaching_recs), 3),
        "n": len(coaching_recs),
        "note": "heuristic: evidence_or_rationale",
    }


def compute_hot_topic_recall(
    aggregated: dict[str, Any], expected_topics: list[str]
) -> dict[str, Any]:
    """Fas 4.3 KPI: recall of hot topics vs an expected topic list (set overlap)."""
    produced = {
        ht.get("topic", "").lower()
        for ht in aggregated.get("hot_topics", [])
        if isinstance(ht, dict)
    }
    # Also accept topic strings nested under insights
    for key in ("topics", "top_topics"):
        for item in aggregated.get(key, []) or []:
            if isinstance(item, str):
                produced.add(item.lower())
            elif isinstance(item, dict) and item.get("topic"):
                produced.add(str(item["topic"]).lower())
    gold = {t.lower() for t in expected_topics}
    if not gold:
        return {"recall": 0.0, "n_gold": 0, "n_produced": len(produced)}
    hit = len(produced & gold)
    return {"recall": round(hit / len(gold), 3), "n_gold": len(gold), "n_produced": len(produced)}


def compute_pii_redaction_coverage(
    pii_log: dict[str, Any] | None, expected_pii_types: list[str] | None = None
) -> dict[str, Any]:
    """Fas 4.4.1 KPI: coverage of PII types caught by the redactor."""
    if not pii_log or not pii_log.get("events"):
        return {"coverage": 0.0, "n_events": 0}
    found_types = {e.get("type") for e in pii_log.get("events", []) if isinstance(e, dict)}
    if not expected_pii_types:
        return {"coverage": 1.0 if found_types else 0.0, "n_events": len(pii_log.get("events", []))}
    gold = set(expected_pii_types)
    hit = len(found_types & gold)
    return {
        "coverage": round(hit / len(gold), 3) if gold else 0.0,
        "n_events": len(pii_log.get("events", [])),
        "found_types": sorted(t for t in found_types if t),
    }


def compute_alert_trigger_rate(alerts: list[dict[str, Any]], total_calls: int) -> dict[str, Any]:
    """Fas 4.4.2 KPI: fraction of calls that generated alerts, plus severity breakdown."""
    if not total_calls:
        return {"trigger_rate": 0.0, "n_alerts": 0}
    n_alerts = len(alerts)
    by_sev: dict[str, int] = {}
    for a in alerts:
        sev = str(a.get("severity", "medium"))
        by_sev[sev] = by_sev.get(sev, 0) + 1
    return {
        "trigger_rate": round(n_alerts / total_calls, 3),
        "n_alerts": n_alerts,
        "by_severity": by_sev,
    }


def compute_cache_hit_rate(cache_hits: int, total_queries: int) -> dict[str, Any]:
    """Fas 4.5.1 KPI: cache effectiveness for pre-computed aggregates."""
    if not total_queries:
        return {"hit_rate": 0.0, "total_queries": 0}
    return {
        "hit_rate": round(cache_hits / total_queries, 3),
        "total_queries": total_queries,
        "cache_hits": cache_hits,
    }
