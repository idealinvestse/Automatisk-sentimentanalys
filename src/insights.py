"""Root cause analysis and key insights for Swedish call center conversations.

Combines diarization, intent, topic, and sentiment to produce
actionable insights about call center performance.

Usage:
    from src.insights import InsightsEngine
    ie = InsightsEngine()
    report = ie.analyze(transcript, diarization, intent_results, sentiment_results, topics)
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RootCause:
    """A root cause finding."""

    issue: str
    evidence: list[str] = field(default_factory=list)
    severity: str = "medium"  # "low", "medium", "high", "critical"
    affected_segments: list[int] = field(default_factory=list)


@dataclass
class AgentMetrics:
    """Per-agent performance metrics."""

    speaker_id: str
    total_segments: int = 0
    positive_ratio: float = 0.0
    negative_ratio: float = 0.0
    avg_response_length: float = 0.0
    resolution_rate: float = 0.0


@dataclass
class InsightsReport:
    """Complete insights report."""

    root_causes: list[RootCause] = field(default_factory=list)
    agent_metrics: list[AgentMetrics] = field(default_factory=list)
    key_findings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    risk_alerts: list[str] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_causes": [
                {"issue": rc.issue, "evidence": rc.evidence, "severity": rc.severity}
                for rc in self.root_causes
            ],
            "agent_metrics": [
                {
                    "speaker_id": am.speaker_id,
                    "total_segments": am.total_segments,
                    "positive_ratio": round(am.positive_ratio, 3),
                    "negative_ratio": round(am.negative_ratio, 3),
                    "avg_response_length": round(am.avg_response_length, 1),
                    "resolution_rate": round(am.resolution_rate, 3),
                }
                for am in self.agent_metrics
            ],
            "key_findings": self.key_findings,
            "recommendations": self.recommendations,
            "risk_alerts": self.risk_alerts,
            "timestamp": self.timestamp,
        }


class InsightsEngine:
    """Generate insights from call center conversation data.

    Combines multiple analysis dimensions to produce root cause analysis,
    agent performance metrics, and actionable recommendations.
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(
        self,
        segments: list[dict[str, Any]],
        intent_results: list[tuple[str, float]] | None = None,
        sentiment_results: list[dict[str, Any]] | None = None,
        topics: list[Any] | None = None,
    ) -> InsightsReport:
        """Run full insights analysis on a call.

        Args:
            segments: ASR segments with optional speaker labels.
            intent_results: Per-segment intent classifications.
            sentiment_results: Per-segment sentiment results.
            topics: Topic modeling results.

        Returns:
            InsightsReport with root causes, metrics, findings, and recommendations.
        """
        root_causes = self._find_root_causes(segments, intent_results, sentiment_results)
        agent_metrics = self._compute_agent_metrics(segments, sentiment_results)
        key_findings = self._extract_key_findings(
            segments, intent_results, sentiment_results, topics
        )
        recommendations = self._generate_recommendations(root_causes, sentiment_results)
        risk_alerts = self._detect_risks(sentiment_results, intent_results)

        return InsightsReport(
            root_causes=root_causes,
            agent_metrics=agent_metrics,
            key_findings=key_findings,
            recommendations=recommendations,
            risk_alerts=risk_alerts,
            timestamp=datetime.now(UTC).isoformat(),
        )

    # ------------------------------------------------------------------
    # Root cause analysis
    # ------------------------------------------------------------------
    def _find_root_causes(
        self,
        segments: list[dict[str, Any]],
        intent_results: list[tuple[str, float]] | None,
        sentiment_results: list[dict[str, Any]] | None,
    ) -> list[RootCause]:
        """Identify root causes of customer issues."""
        causes: list[RootCause] = []

        # Check for repeated complaints
        if intent_results:
            complaint_count = sum(1 for i, _ in intent_results if i == "complaint")
            if complaint_count >= 3:
                causes.append(
                    RootCause(
                        issue="Upprepade klagomål under samtalet",
                        evidence=[f"{complaint_count} segment klassificerade som klagomål"],
                        severity="high",
                    )
                )

        # Check for billing issues
        if intent_results:
            billing_count = sum(1 for i, _ in intent_results if i == "billing_inquiry")
            if billing_count >= 2:
                causes.append(
                    RootCause(
                        issue="Faktureringsrelaterat problem",
                        evidence=[f"{billing_count} segment om fakturering"],
                        severity="medium",
                    )
                )

        # Check for technical issues combined with negative sentiment
        if intent_results and sentiment_results:
            if len(intent_results) != len(sentiment_results):
                logger.warning(
                    "Intent results (%d) and sentiment results (%d) length mismatch. "
                    "Skipping combined root cause analysis.",
                    len(intent_results),
                    len(sentiment_results),
                )
            else:
                tech_neg = 0
                for (intent, _), sent in zip(intent_results, sentiment_results, strict=True):
                    if intent == "technical_support" and sent.get("label") == "negativ":
                        tech_neg += 1
                if tech_neg >= 2:
                    causes.append(
                        RootCause(
                            issue="Tekniskt problem som orsakar missnöje",
                            evidence=[f"{tech_neg} negativa segment om tekniska problem"],
                            severity="high",
                        )
                    )
        # Check for agent-related issues
        if segments:
            agent_segments = [s for s in segments if s.get("speaker", "") not in ("", "UNKNOWN")]
            if agent_segments:
                speakers = Counter(s.get("speaker") for s in agent_segments)
                if len(speakers) > 2:
                    causes.append(
                        RootCause(
                            issue="Många inblandade agenter – risk för rundgång",
                            evidence=[f"{len(speakers)} olika talare identifierade"],
                            severity="medium",
                        )
                    )

        return causes

    # ------------------------------------------------------------------
    # Agent metrics
    # ------------------------------------------------------------------
    def _compute_agent_metrics(
        self,
        segments: list[dict[str, Any]],
        sentiment_results: list[dict[str, Any]] | None,
    ) -> list[AgentMetrics]:
        """Compute per-agent performance metrics."""
        speaker_segments: dict[str, list[dict[str, Any]]] = {}
        for seg in segments:
            speaker = seg.get("speaker", "UNKNOWN")
            if speaker != "UNKNOWN":
                speaker_segments.setdefault(speaker, []).append(seg)

        metrics: list[AgentMetrics] = []
        for speaker, segs in speaker_segments.items():
            am = AgentMetrics(speaker_id=speaker, total_segments=len(segs))

            # Average response length
            lengths = [len(s.get("text", "")) for s in segs]
            am.avg_response_length = sum(lengths) / max(1, len(lengths))

            # Sentiment ratios if available
            if sentiment_results and len(sentiment_results) == len(segments):
                sent_labels = [
                    sentiment_results[i].get("label", "neutral")
                    for i in range(len(segments))
                    if segments[i].get("speaker") == speaker
                ]
                if sent_labels:
                    am.positive_ratio = sent_labels.count("positiv") / len(sent_labels)
                    am.negative_ratio = sent_labels.count("negativ") / len(sent_labels)

            metrics.append(am)

        return metrics

    # ------------------------------------------------------------------
    # Key findings
    # ------------------------------------------------------------------
    def _extract_key_findings(
        self,
        segments: list[dict[str, Any]],
        intent_results: list[tuple[str, float]] | None,
        sentiment_results: list[dict[str, Any]] | None,
        topics: list[Any] | None,
    ) -> list[str]:
        """Extract key findings from the analysis."""
        findings: list[str] = []

        if sentiment_results:
            labels = [r.get("label", "neutral") for r in sentiment_results]
            pos_ratio = labels.count("positiv") / max(1, len(labels))
            neg_ratio = labels.count("negativ") / max(1, len(labels))
            findings.append(
                f"Sentimentfördelning: {pos_ratio:.0%} positiva, {neg_ratio:.0%} negativa"
            )

        if intent_results:
            top_intents = Counter(i for i, _ in intent_results).most_common(3)
            findings.append(
                f"Vanligaste ärendetyper: {', '.join(f'{i} ({c})' for i, c in top_intents)}"
            )

        if topics:
            topic_names = [getattr(t, "name", str(t)) for t in topics]
            if topic_names:
                findings.append(f"Huvudämnen: {', '.join(topic_names[:5])}")

        return findings

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------
    def _generate_recommendations(
        self,
        root_causes: list[RootCause],
        sentiment_results: list[dict[str, Any]] | None,
    ) -> list[str]:
        """Generate actionable recommendations."""
        recs: list[str] = []

        for rc in root_causes:
            if "fakturer" in rc.issue.lower():
                recs.append("Se över faktureringsrutiner och informationsmaterial till kund")
            if "tekniskt" in rc.issue.lower():
                recs.append("Prioritera teknisk felsökning och eskalera till teknikavdelning")
            if "klagomål" in rc.issue.lower():
                recs.append("Överväg kompensation och personlig uppföljning")
            if "agent" in rc.issue.lower() or "rundgång" in rc.issue.lower():
                recs.append("Säkerställ tydlig ägaröverlämning mellan agenter")

        if sentiment_results:
            neg_count = sum(1 for r in sentiment_results if r.get("label") == "negativ")
            if neg_count > len(sentiment_results) * 0.3:
                recs.append("Hög andel negativitet – överväg proaktiv outreach till kund")

        return recs[:5]

    # ------------------------------------------------------------------
    # Risk detection
    # ------------------------------------------------------------------
    def _detect_risks(
        self,
        sentiment_results: list[dict[str, Any]] | None,
        intent_results: list[tuple[str, float]] | None,
    ) -> list[str]:
        """Detect risk alerts (churn, escalation)."""
        alerts: list[str] = []

        if sentiment_results:
            neg_count = sum(1 for r in sentiment_results if r.get("label") == "negativ")
            if neg_count > len(sentiment_results) * 0.5:
                alerts.append("HÖG CHURN-RISK: Över 50% negativa segment")

        if intent_results:
            complaint_count = sum(1 for i, _ in intent_results if i == "complaint")
            cancellation_count = sum(1 for i, _ in intent_results if i == "cancellation")
            if complaint_count + cancellation_count >= 3:
                alerts.append("HÖG ESCALATION-RISK: Flera klagomål/uppsägningssignaler")

        return alerts


__all__ = ["InsightsEngine", "InsightsReport", "RootCause", "AgentMetrics"]
