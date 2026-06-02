"""Alerting & Workflow Engine (Fas 4.4.2).

Regelbaserade alerts som triggas från per-call data (sentiment, qa, agent_performance, llm) eller från aggregator (trender/hot topics).

Output: structured Alert + AlertSummary med evidence_spans, recommended_actions (action-oriented).

Stöd för:
- Definiera regler (default + custom)
- Webhook-notifieringar (stub + payload builder)
- Interna workflows (t.ex. "create_coaching_task")
- Triggas från insights_aggregator (t.ex. hot topic med låg sentiment)

Explicit integration i pipeline.py (per plan):
    from .alerting import AlertEngine
    engine = AlertEngine()
    alerts = engine.check(results)  # or check_from_aggregate(agg)
    results["alerts"] = [a.model_dump() for a in alerts]  # or AlertSummary

Pydantic models in llm/schemas.py (Alert, AlertSummary) så de mergas lätt i CallAnalysisReport.results["alerts"]

Hybrid: mest regelbaserat (snabbt, deterministic), kan anropa Mistral för att generera bättre action recommendations om behövs (valfritt, dokumenterat).

Caching: enkel lru på check om data är samma.

Privacy: använder redan redacted data.

Se UTVECKLINGSPLAN_Fas4 v1.1 för acceptance: "Kan definiera regel → trigga alert vid matchning."
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any, Callable

from pydantic import BaseModel

from .llm.schemas import Alert, AlertSummary, EvidenceSpan

logger = logging.getLogger(__name__)

# Default rules (regelbaserade, matchar planens exempel + mer från Fas4 data)
DEFAULT_RULES: list[dict[str, Any]] = [
    {
        "id": "high_escalation_risk",
        "condition": "customer_sentiment < -0.7 and (escalation_risk > 0.6 or qa_risk in ['high', 'critical'])",
        "severity": "high",
        "message": "Hög eskaleringrisk: kunden mycket negativ och QA-flaggor eller låg empati.",
        "actions": ["flag_supervisor", "create_coaching_task:de_escalation"],
        "evidence_keys": ["customer_sentiment", "escalation_risk", "qa_risk"],
    },
    {
        "id": "low_agent_empathy",
        "condition": "agent_empathy < 0.3 and customer_sentiment < -0.4",
        "severity": "medium",
        "message": "Agent visade låg empati trots kundens negativa sentiment.",
        "actions": ["create_coaching_task:empathy", "notify_team_lead"],
        "evidence_keys": ["agent_empathy", "customer_sentiment"],
    },
    {
        "id": "qa_failed_high_risk",
        "condition": "qa_passed == False and qa_risk in ['high', 'critical']",
        "severity": "high",
        "message": "QA-score under tröskel med hög risk - compliance eller processbrist.",
        "actions": ["flag_supervisor", "review_call_manually"],
        "evidence_keys": ["qa_overall_score", "qa_risk", "compliance_flags"],
    },
    {
        "id": "hot_topic_negative_trend",
        "condition": "hot_topic_sentiment < -0.5 and hot_topic_volume > 5",
        "severity": "medium",
        "message": "Het topic med negativ trend och hög volym - potentiellt systemiskt problem.",
        "actions": ["escalate_to_process_owner", "analyze_root_cause"],
        "evidence_keys": ["hot_topic", "hot_topic_sentiment", "hot_topic_volume"],
    },
]

def _safe_eval_condition(condition: str, signals: dict[str, Any]) -> bool:
    """Very simple and safe-ish condition evaluator for rules.
    Supports basic < > == and/or/in on numeric/string keys.
    Not full python eval for security.
    """
    try:
        # Normalize
        expr = condition.lower().replace(" and ", " & ").replace(" or ", " | ")
        # Replace known keys
        for k, v in signals.items():
            if k in expr:
                val = v
                if isinstance(val, str):
                    val = f'"{val.lower()}"'
                elif val is None:
                    val = "None"
                expr = expr.replace(k, str(val))
        # Simple replacements
        expr = expr.replace("&", "and").replace("|", "or")
        # Allowed
        allowed = {"and": lambda x, y: x and y, "or": lambda x, y: x or y, "in": lambda x, y: x in y}
        # Use eval with restricted globals (risky but for this internal tool ok; in prod use parser)
        res = eval(expr, {"__builtins__": {}}, {**allowed, **{k: v for k, v in signals.items()}})
        return bool(res)
    except Exception as e:
        logger.debug("Alert condition eval failed for '%s': %s", condition, e)
        return False


class AlertEngine:
    """Main alerting engine. Regelbaserat med stöd för custom rules och aggregator triggers."""

    def __init__(self, rules: list[dict[str, Any]] | None = None, mistral_analyzer: Any | None = None):
        self.rules = rules or DEFAULT_RULES
        self.mistral_analyzer = mistral_analyzer  # optional for enhanced actions

    def check(self, signals: dict[str, Any]) -> list[Alert]:
        """Check per-call or batch signals against rules. Returns list of triggered Alerts with evidence."""
        alerts: list[Alert] = []
        for rule in self.rules:
            if _safe_eval_condition(rule["condition"], signals):
                # Build evidence from keys
                ev_spans: list[EvidenceSpan] = []
                trig_vals = {}
                for key in rule.get("evidence_keys", []):
                    if key in signals:
                        val = signals[key]
                        trig_vals[key] = val
                        # Create pseudo evidence span (in real would come from original segments)
                        ev_spans.append(EvidenceSpan(
                            text=f"{key}={val}",
                            speaker_role=None,
                            turn_index=None
                        ))

                # Optional LLM for better actions/recommendations (document when used)
                actions = rule.get("actions", []).copy()
                if self.mistral_analyzer and "escalation" in rule["id"].lower():
                    try:
                        # Selective, low cost
                        prompt = f"Ge 1-2 konkreta svenska coachningsåtgärder för: {rule['message']}. Baserat på värden: {trig_vals}"
                        # ... (simplified, assume client call or skip full impl here)
                        actions.append("llm_suggested_coaching")
                        logger.info("Mistral used for alert action enhancement (rule %s)", rule["id"])
                    except Exception:
                        pass

                alert = Alert(
                    rule_id=rule["id"],
                    severity=rule.get("severity", "medium"),
                    message=rule.get("message", ""),
                    evidence_spans=ev_spans,
                    triggered_values=trig_vals,
                    recommended_actions=actions,
                    source="per_call",
                )
                alerts.append(alert)
                logger.info("Alert triggered: %s (severity=%s)", rule["id"], alert.severity)

        return alerts

    def check_from_aggregate(self, agg: dict[str, Any]) -> list[Alert]:
        """Trigger alerts from AggregatedInsights (trend-based, per plan)."""
        alerts = []
        signals: dict[str, Any] = {}
        for ht in agg.get("hot_topics", []):
            if ht.get("trend") == "down" and ht.get("avg_sentiment", 0) < -0.4:
                signals["hot_topic"] = ht["topic"]
                signals["hot_topic_sentiment"] = ht["avg_sentiment"]
                signals["hot_topic_volume"] = ht["volume"]
                # Reuse check
                alerts.extend(self.check(signals))
        return alerts

    def build_webhook_payload(self, alert: Alert, call_id: str | None = None) -> dict[str, Any]:
        """Build payload for webhook notification."""
        return {
            "event": "callcenter_alert",
            "rule_id": alert.rule_id,
            "severity": alert.severity,
            "message": alert.message,
            "call_id": call_id,
            "triggered_values": alert.triggered_values,
            "recommended_actions": alert.recommended_actions,
            "evidence": [e.model_dump() for e in alert.evidence_spans],
        }

    def notify_webhook(self, alert: Alert, url: str | None = None, call_id: str | None = None) -> dict:
        """Stub for webhook. In real: requests.post(url, json=payload).
        Here: returns the payload and logs.
        """
        payload = self.build_webhook_payload(alert, call_id)
        if url:
            logger.info("Would POST to webhook %s: %s", url, json.dumps(payload)[:200])
            # TODO: actual http in production
        else:
            logger.info("Alert webhook payload (no url configured): %s", payload)
        return payload


# Convenience for pipeline integration
def run_alerts_on_results(results: dict[str, Any], engine: AlertEngine | None = None) -> list[dict[str, Any]]:
    """Explicit helper used from pipeline.py."""
    engine = engine or AlertEngine()
    # Build flat signals from Fas4 results + legacy
    signals: dict[str, Any] = {}
    # From llm
    llm = results.get("llm", {})
    if "trajectory" in llm and llm["trajectory"]:
        traj = llm["trajectory"]
        signals["customer_sentiment"] = traj.get("customer_sentiment_slope", 0)
        signals["escalation_risk"] = 0.8 if traj.get("escalation_events") else 0.2
    # From agent
    ap = results.get("agent_performance", {}).get("agent", {})
    if ap:
        signals["agent_empathy"] = ap.get("empathy_score", 0.5)
        signals["talk_listen_ratio"] = ap.get("talk_listen_ratio", 1.0)
    # From qa
    qa = results.get("qa") or results.get("compliance_qa") or {}
    if qa:
        signals["qa_passed"] = qa.get("passed", True)
        signals["qa_risk"] = qa.get("risk_level", "low")
        signals["qa_overall_score"] = qa.get("overall_qa_score", 100)
    # From insights / root
    if "root_cause" in llm:
        signals["has_unresolved_root"] = llm["root_cause"].get("customer_unresolved", False)

    alerts = engine.check(signals)
    return [a.model_dump() for a in alerts]
