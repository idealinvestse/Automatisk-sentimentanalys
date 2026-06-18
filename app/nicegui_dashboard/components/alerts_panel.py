"""Alerts & Actions panel with severity and mark-as-handled stub.

Fas 3 – docs/GROK_BUILD_PLAN_FAS1-3.md
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nicegui import ui

from app.nicegui_dashboard.services.fas4_data import (
    active_alerts,
    alert_dedup_key,
    format_evidence_spans,
    severity_color,
)
from app.nicegui_dashboard.state import DashboardState
from app.services.data_services import filter_reports


def count_active_alerts(state: DashboardState) -> int:
    """Active (non-dismissed) alert count for header badge."""
    reports = filter_reports(state.reports, state.filters)
    return len(active_alerts(reports, state.dismissed_alert_keys))


def render_alerts_panel(
    state: DashboardState,
    *,
    on_call_select: Callable[[str], None] | None = None,
    on_dismiss_change: Callable[[], None] | None = None,
    compact: bool = False,
) -> Callable[[], None]:
    """Render alerts list; returns refresh callback."""

    @ui.refreshable
    def alerts_section() -> None:
        reports = filter_reports(state.reports, state.filters)
        alerts = active_alerts(reports, state.dismissed_alert_keys)

        title = "🚨 Alerts & Actions" if not compact else "Alerts"
        ui.label(title).classes("text-subtitle1 q-mb-sm" if not compact else "text-subtitle2")

        if not alerts:
            ui.label("Inga aktiva alerts.").classes("text-caption text-positive")
            return

        ui.label(f"{len(alerts)} aktiva alerts").classes("text-caption q-mb-xs")

        for alert in alerts[:20]:
            if not isinstance(alert, dict):
                continue
            key = alert_dedup_key(alert)
            severity = str(alert.get("severity", "info"))
            call_id = alert.get("call_id", "—")
            message = alert.get("message", alert.get("rule_id", "Alert"))
            actions = alert.get("recommended_actions") or []

            with ui.card().classes("w-full q-mb-sm"):
                with ui.row().classes("w-full items-start justify-between gap-2"):
                    with ui.column().classes("flex-grow"):
                        with ui.row().classes("items-center gap-2"):
                            ui.chip(severity, color=severity_color(severity))
                            ui.label(str(call_id)).classes("text-caption text-grey")
                        ui.label(str(message)).classes("text-body2")
                        ev = format_evidence_spans(alert.get("evidence_spans") or [])
                        if ev != "—":
                            ui.label(ev).classes("text-caption text-grey")
                        if actions:
                            ui.label(
                                "Åtgärder: " + ", ".join(str(a) for a in actions[:4])
                            ).classes("text-caption")
                    with ui.column().classes("items-end gap-1"):
                        if on_call_select and call_id and call_id != "—":
                            ui.button(
                                "Öppna samtal",
                                on_click=lambda c=str(call_id): on_call_select(c),
                            ).props("flat dense")
                        ui.button(
                            "Markera hanterad",
                            on_click=lambda k=key: _dismiss(k),
                        ).props("flat dense color=grey")

    def _dismiss(key: str) -> None:
        if key not in state.dismissed_alert_keys:
            state.dismissed_alert_keys.append(key)
            ui.notify("Alert markerad som hanterad", type="positive")
        alerts_section.refresh()
        if on_dismiss_change:
            on_dismiss_change()

    alerts_section()
    return alerts_section.refresh