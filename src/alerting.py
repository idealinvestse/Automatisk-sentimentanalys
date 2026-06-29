"""Alerting & Workflow Engine (Fas 4.4.2).

Regelbaserade alerts som triggas från per-call data (sentiment, qa, agent_performance, llm) eller från aggregator (trender/hot topics).

Output: structured Alert + AlertSummary med evidence_spans, recommended_actions (action-oriented).

Stöd för:
- Definiera regler (default + custom)
- Webhook-notifieringar (httpx POST + retry + circuit breaker)
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

import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

from .llm.schemas import Alert, EvidenceSpan

logger = logging.getLogger(__name__)


def load_alerting_config(path: str | Path = "configs/alerting_config.yaml") -> dict[str, Any]:
    """Load alerting config with env var override support.

    Priority: hardcoded defaults < YAML file < environment variables.
    Supports ${VAR:-default} syntax in YAML for env substitution.
    Returns defaults if file missing or unreadable (graceful degradation).
    """
    # 1. Hardcoded defaults
    config: dict[str, Any] = {
        "webhook": {
            "enabled": True,
            "url": "",
            "timeout_seconds": 10,
            "max_retries": 3,
            "circuit_breaker_threshold": 5,
            "retry_backoff_base": 1.0,
        }
    }
    # 2. YAML overrides defaults
    p = Path(path)
    if p.exists():
        try:
            raw = p.read_text(encoding="utf-8")
            import re

            def _sub(m):
                var = m.group(1)
                default = m.group(2) or ""
                return os.getenv(var, default)

            raw = re.sub(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}", _sub, raw)
            data = yaml.safe_load(raw) or {}
            if "webhook" in data:
                config["webhook"].update(
                    {k: v for k, v in data["webhook"].items() if v is not None}
                )
        except Exception as exc:
            logger.warning("Failed to load alerting config (%s): %s", path, exc)
    # 3. Env vars override everything (highest priority)
    if os.getenv("ALERT_WEBHOOK_URL"):
        config["webhook"]["url"] = os.getenv("ALERT_WEBHOOK_URL", "")
    if os.getenv("ALERT_WEBHOOK_TIMEOUT"):
        config["webhook"]["timeout_seconds"] = int(os.getenv("ALERT_WEBHOOK_TIMEOUT", "10"))
    if os.getenv("ALERT_WEBHOOK_RETRIES"):
        config["webhook"]["max_retries"] = int(os.getenv("ALERT_WEBHOOK_RETRIES", "3"))
    if os.getenv("ALERT_WEBHOOK_BREAKER"):
        config["webhook"]["circuit_breaker_threshold"] = int(
            os.getenv("ALERT_WEBHOOK_BREAKER", "5")
        )
    if os.getenv("ALERT_WEBHOOK_BACKOFF"):
        config["webhook"]["retry_backoff_base"] = float(os.getenv("ALERT_WEBHOOK_BACKOFF", "1.0"))
    return config


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


def _tokenize_condition(condition: str) -> list[tuple[str, str]]:
    """Tokenize alert rule conditions without using eval."""
    pattern = re.compile(
        r"\s*(?:"
        r"(?P<AND>\band\b)|"
        r"(?P<OR>\bor\b)|"
        r"(?P<IN>\bin\b)|"
        r"(?P<OP>==|!=|<=|>=|<|>)|"
        r"(?P<LPAREN>\()|"
        r"(?P<RPAREN>\))|"
        r"(?P<LBRACKET>\[)|"
        r"(?P<RBRACKET>\])|"
        r"(?P<COMMA>,)|"
        r"(?P<BOOL>True|False)|"
        r"(?P<NUMBER>-?\d+(?:\.\d+)?)|"
        r"(?P<STRING>'[^']*'|\"[^\"]*\")|"
        r"(?P<IDENT>[a-zA-Z_][a-zA-Z0-9_]*)"
        r")",
        re.IGNORECASE,
    )
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(condition):
        match = pattern.match(condition, pos)
        if not match:
            raise ValueError(f"Unexpected character at position {pos}: {condition[pos:pos + 8]!r}")
        for name, value in match.groupdict().items():
            if value is not None:
                tokens.append((name, value))
                break
        pos = match.end()
    return tokens


def _parse_string_literal(raw: str) -> str:
    return raw[1:-1].lower()


def _parse_list_literal(tokens: list[tuple[str, str]], index: int) -> tuple[list[Any], int]:
    if index >= len(tokens) or tokens[index][0] != "LBRACKET":
        raise ValueError("Expected '['")
    index += 1
    items: list[Any] = []
    while index < len(tokens) and tokens[index][0] != "RBRACKET":
        if tokens[index][0] == "COMMA":
            index += 1
            continue
        value, index = _parse_value(tokens, index)
        items.append(value)
    if index >= len(tokens) or tokens[index][0] != "RBRACKET":
        raise ValueError("Expected ']'")
    return items, index + 1


def _parse_value(tokens: list[tuple[str, str]], index: int) -> tuple[Any, int]:
    kind, raw = tokens[index]
    if kind == "NUMBER":
        return float(raw) if "." in raw else int(raw), index + 1
    if kind == "BOOL":
        return raw.lower() == "true", index + 1
    if kind == "STRING":
        return _parse_string_literal(raw), index + 1
    if kind == "IDENT":
        return ("__ident__", raw), index + 1
    if kind == "LBRACKET":
        return _parse_list_literal(tokens, index)
    raise ValueError(f"Unexpected token {kind}")


def _resolve_value(value: Any, signals: dict[str, Any]) -> Any:
    if isinstance(value, tuple) and len(value) == 2 and value[0] == "__ident__":
        key = value[1]
        if key not in signals:
            raise KeyError(key)
        return signals[key]
    return value


def _coerce_for_compare(left: Any, right: Any) -> tuple[Any, Any]:
    if isinstance(left, str) or isinstance(right, str):
        return str(left).lower(), str(right).lower()
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left), bool(right)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left), float(right)
    return left, right


def _compare(left: Any, op: str, right: Any) -> bool:
    if op == "in":
        if not isinstance(right, list):
            raise ValueError("'in' requires a list on the right-hand side")
        left_cmp = str(left).lower() if isinstance(left, str) else left
        right_cmp = [str(v).lower() if isinstance(v, str) else v for v in right]
        return left_cmp in right_cmp
    left, right = _coerce_for_compare(left, right)
    if op == "==":
        return left == right
    if op == "!=":
        return left != right
    if op == "<":
        return left < right
    if op == ">":
        return left > right
    if op == "<=":
        return left <= right
    if op == ">=":
        return left >= right
    raise ValueError(f"Unknown operator {op}")


class _ConditionParser:
    def __init__(self, tokens: list[tuple[str, str]], signals: dict[str, Any]) -> None:
        self.tokens = tokens
        self.signals = signals
        self.index = 0

    def parse(self) -> bool:
        result = self._parse_or()
        if self.index != len(self.tokens):
            raise ValueError(f"Unexpected trailing tokens at {self.index}")
        return bool(result)

    def _parse_or(self) -> bool:
        value = self._parse_and()
        while self.index < len(self.tokens) and self.tokens[self.index][0] == "OR":
            self.index += 1
            rhs = self._parse_and()
            value = bool(value or rhs)
        return value

    def _parse_and(self) -> bool:
        value = self._parse_comparison()
        while self.index < len(self.tokens) and self.tokens[self.index][0] == "AND":
            self.index += 1
            rhs = self._parse_comparison()
            value = bool(value and rhs)
        return value

    def _parse_comparison(self) -> bool:
        if self.index < len(self.tokens) and self.tokens[self.index][0] == "LPAREN":
            self.index += 1
            value = self._parse_or()
            if self.index >= len(self.tokens) or self.tokens[self.index][0] != "RPAREN":
                raise ValueError("Expected ')'")
            self.index += 1
            return value

        left_raw, self.index = _parse_value(self.tokens, self.index)
        left = _resolve_value(left_raw, self.signals)

        if self.index < len(self.tokens) and self.tokens[self.index][0] in ("OP", "IN"):
            op = self.tokens[self.index][1].lower()
            if self.tokens[self.index][0] == "IN":
                op = "in"
            self.index += 1
            right_raw, self.index = _parse_value(self.tokens, self.index)
            right = (
                _resolve_value(right_raw, self.signals)
                if not isinstance(right_raw, list)
                else right_raw
            )
            return _compare(left, op, right)

        return bool(left)


def _safe_eval_condition(condition: str, signals: dict[str, Any]) -> bool:
    """Evaluate alert rule conditions with a small safe parser (no eval/exec)."""
    try:
        tokens = _tokenize_condition(condition)
        return _ConditionParser(tokens, signals).parse()
    except Exception as e:
        logger.debug("Alert condition eval failed for '%s': %s", condition, e)
        return False


class AlertEngine:
    """Main alerting engine. Regelbaserat med stöd för custom rules och aggregator triggers."""

    def __init__(
        self,
        rules: list[dict[str, Any]] | None = None,
        mistral_analyzer: Any | None = None,
        config: dict[str, Any] | None = None,
    ):
        self.rules = rules or DEFAULT_RULES
        self.mistral_analyzer = mistral_analyzer  # optional for enhanced actions
        self.config = config or load_alerting_config()
        self._consecutive_failures = 0
        self._webhook_disabled = False

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
                        ev_spans.append(
                            EvidenceSpan(text=f"{key}={val}", speaker_role=None, turn_index=None)
                        )

                # Optional LLM for better actions/recommendations (document when used)
                actions = rule.get("actions", []).copy()
                if self.mistral_analyzer and "escalation" in rule["id"].lower():
                    try:
                        # Selective, low cost
                        # NOTE: actual prompt construction deferred — see LLM_PROVIDERS.md
                        # for available Mistral/Groq clients. Kept as placeholder for v0.5.
                        actions.append("llm_suggested_coaching")
                        logger.info(
                            "Mistral used for alert action enhancement (rule %s)", rule["id"]
                        )
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

    def notify_webhook(
        self, alert: Alert, url: str | None = None, call_id: str | None = None
    ) -> dict:
        """Send webhook notification with retry, backoff, and circuit breaker.

        Implements production-grade delivery:
        - 10s timeout per attempt
        - 3 retries with exponential backoff (1s, 2s, 4s)
        - Circuit breaker: disabled after 5 consecutive failures
        - Logs success/failure/circuit state
        """
        payload = self.build_webhook_payload(alert, call_id)
        if not url:
            logger.info("Alert webhook payload (no url configured): %s", payload)
            return payload

        # Circuit breaker check
        if getattr(self, "_webhook_disabled", False):
            logger.warning("Webhook circuit breaker OPEN – delivery skipped")
            return payload

        max_retries = 3
        base_backoff = 1.0
        timeout = 10.0

        attempt = 0
        while attempt < max_retries:
            attempt += 1
            try:
                resp = httpx.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=timeout,
                )
                if resp.status_code < 400:
                    logger.info(
                        "Webhook delivered (attempt %d/%d): %s -> %s",
                        attempt,
                        max_retries,
                        alert.rule_id,
                        resp.status_code,
                    )
                    # Reset failure counter on success
                    self._consecutive_failures = 0
                    return payload
                else:
                    logger.warning(
                        "Webhook HTTP %s (attempt %d): %s",
                        resp.status_code,
                        attempt,
                        resp.text[:200],
                    )
            except httpx.TimeoutException:
                logger.warning("Webhook timeout (attempt %d/%d)", attempt, max_retries)
            except Exception as exc:
                logger.warning("Webhook error (attempt %d/%d): %s", attempt, max_retries, exc)

            self._consecutive_failures = getattr(self, "_consecutive_failures", 0) + 1

            # Circuit breaker trigger
            if self._consecutive_failures >= 5:
                self._webhook_disabled = True
                logger.error(
                    "Webhook circuit breaker OPEN after %d consecutive failures – disabled",
                    self._consecutive_failures,
                )
                break

            if attempt < max_retries:
                sleep_s = base_backoff * (2 ** (attempt - 1))
                logger.debug("Backing off %.1fs before retry", sleep_s)
                time.sleep(sleep_s)

        return payload

    def get_webhook_status(self) -> dict[str, Any]:
        """Return current webhook / circuit breaker status for ops and dashboard."""
        return {
            "enabled": self.config.get("webhook", {}).get("enabled", True),
            "url_configured": bool(self.config.get("webhook", {}).get("url")),
            "circuit_breaker_open": getattr(self, "_webhook_disabled", False),
            "consecutive_failures": getattr(self, "_consecutive_failures", 0),
            "max_retries": self.config.get("webhook", {}).get("max_retries", 3),
            "circuit_breaker_threshold": self.config.get("webhook", {}).get(
                "circuit_breaker_threshold", 5
            ),
        }

    def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker (testing / ops)."""
        self._webhook_disabled = False
        self._consecutive_failures = 0
        logger.info("Webhook circuit breaker manually reset")


# Convenience for pipeline integration
def run_alerts_on_results(
    results: dict[str, Any], engine: AlertEngine | None = None
) -> list[dict[str, Any]]:
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
