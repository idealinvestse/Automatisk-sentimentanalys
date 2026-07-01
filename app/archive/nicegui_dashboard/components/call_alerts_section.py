"""Compact per-call alerts section for Samtalsdetalj."""

from __future__ import annotations

from typing import Any

from nicegui import ui

from app.archive.nicegui_dashboard.components.empty_state import render_empty_state
from app.archive.nicegui_dashboard.services.fas4_data import format_evidence_spans, severity_color


def render_call_alerts_section(report: dict[str, Any] | None) -> None:
    """Render alerts from a single report (no dismiss / webhook UI)."""
    if not report:
        return
    alerts = (report.get("results") or {}).get("alerts") or []
    with ui.expansion("Aviseringar", icon="notifications", value=bool(alerts)).classes("w-full"):
        if not alerts:
            render_empty_state(
                icon="notifications_none",
                title="Inga aviseringar",
                hint="Inga alerts triggades för detta samtal.",
            )
            return
        for alert in alerts[:15]:
            if not isinstance(alert, dict):
                continue
            severity = str(alert.get("severity", "info"))
            rule_id = alert.get("rule_id", "")
            message = alert.get("message", rule_id or "Avisering")
            evidence = format_evidence_spans(alert.get("evidence_spans") or [])
            with ui.card().classes("q-mb-sm q-pa-sm"):
                with ui.row().classes("items-center gap-2"):
                    ui.badge(severity.upper(), color=severity_color(severity))
                    if rule_id:
                        ui.label(str(rule_id)).classes("text-caption text-grey")
                ui.label(str(message)).classes("text-body2")
                if evidence:
                    ui.label(evidence).classes("text-caption text-grey")
