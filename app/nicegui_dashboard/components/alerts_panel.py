"""Alerts & Actions panel with improved mark-as-handled UX and basic status.

Fas 4 polish – better separation of active vs handled alerts + webhook health hint.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
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

_SEVERITY_FILTER_OPTIONS = {
    "": "Alla",
    "critical": "kritisk",
    "high": "hög",
    "medium": "medium",
    "low": "låg",
}


def _alert_matches_severity(alert: dict[str, Any], severity_filter: str) -> bool:
    if not severity_filter:
        return True
    sev = str(alert.get("severity", "")).lower()
    if severity_filter == "low":
        return sev in ("low", "info")
    return sev == severity_filter


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
    """Render alerts list with improved handled section and status."""
    severity_filter: dict[str, str] = {"value": ""}

    @ui.refreshable
    async def alerts_section() -> None:
        reports = filter_reports(state.reports, state.filters)

        if state.api_client and not getattr(state, "alerting_status", None):
            try:
                status = await state.api_client.get_alerting_status()
                state.alerting_status = status.get("webhook", {})
            except Exception:
                state.alerting_status = {"circuit_breaker_open": False}

        active = active_alerts(reports, state.dismissed_alert_keys)
        filtered_active = [
            a for a in active if _alert_matches_severity(a, severity_filter["value"])
        ]

        title = "🚨 Alerts & Actions" if not compact else "Alerts"
        ui.label(title).classes("text-subtitle1 q-mb-sm" if not compact else "text-subtitle2")

        with ui.row().classes("w-full items-center gap-2 flex-wrap q-mb-sm"):
            ui.select(
                options=_SEVERITY_FILTER_OPTIONS,
                value=severity_filter["value"],
                label="Allvarlighetsgrad",
                on_change=lambda e: (
                    severity_filter.update({"value": e.value or ""}),
                    alerts_section.refresh(),
                ),
            ).classes("min-w-40").props("dense")
            if filtered_active:
                ui.button(
                    "Exportera JSON",
                    icon="download",
                    on_click=lambda: _export_active_alerts(filtered_active),
                ).props("outline dense")

        status = getattr(state, "alerting_status", {}) or {}
        is_open = status.get("circuit_breaker_open", False)
        cb_color = "negative" if is_open else "positive"
        status_text = "ÖPPEN" if is_open else "STÄNGD"

        with ui.row().classes("items-center gap-2 q-mb-sm"):
            ui.chip("Webhook (avisering)", color="primary").props("dense")
            ui.chip(f"Kretsbrytare: {status_text}", color=cb_color).props("dense")
            if status:
                failures = status.get("consecutive_failures", 0)
                threshold = status.get("circuit_breaker_threshold", 5)
                ui.label(f"({failures}/{threshold} failures)").classes("text-caption text-grey")

        if not filtered_active:
            if active and severity_filter["value"]:
                ui.label("Inga alerts matchar valt filter.").classes(
                    "text-caption text-warning q-mb-md"
                )
            else:
                ui.label("Inga aktiva alerts just nu.").classes(
                    "text-caption text-positive q-mb-md"
                )
        else:
            ui.label(f"{len(filtered_active)} aktiva alerts").classes("text-caption q-mb-xs")

            if len(filtered_active) > 5:
                table_rows = []
                for alert in filtered_active[:50]:
                    if not isinstance(alert, dict):
                        continue
                    actions = alert.get("recommended_actions") or []
                    table_rows.append(
                        {
                            "key": alert_dedup_key(alert),
                            "severity": str(alert.get("severity", "info")),
                            "rule_id": str(alert.get("rule_id", "—")),
                            "call_id": str(alert.get("call_id", "—")),
                            "message": str(
                                alert.get("message", alert.get("rule_id", "Alert"))
                            )[:120],
                            "actions": ", ".join(str(a) for a in actions[:3]) or "—",
                        }
                    )
                alert_table = ui.table(
                    columns=[
                        {"name": "severity", "label": "Nivå", "field": "severity"},
                        {"name": "rule_id", "label": "Regel", "field": "rule_id"},
                        {"name": "call_id", "label": "Samtal", "field": "call_id"},
                        {"name": "message", "label": "Meddelande", "field": "message"},
                        {"name": "actions", "label": "Åtgärder", "field": "actions"},
                    ],
                    rows=table_rows,
                    row_key="key",
                    selection="single",
                ).classes("w-full q-mb-sm")

                def _alert_row_click(e: Any) -> None:
                    row = e.args[1] if len(e.args) > 1 else e.args[0]
                    if not isinstance(row, dict):
                        return
                    cid = row.get("call_id")
                    if on_call_select and cid and cid != "—":
                        on_call_select(str(cid))

                alert_table.on("rowClick", _alert_row_click)

                def _dismiss_selected() -> None:
                    selected = alert_table.selected
                    if not selected:
                        ui.notify("Välj en alert i tabellen", type="warning")
                        return
                    for row in selected:
                        _dismiss(str(row.get("key", "")))

                ui.button(
                    "Markera vald som hanterad",
                    icon="check",
                    on_click=_dismiss_selected,
                ).props("flat dense color=positive")
            else:
                for alert in filtered_active[:25]:
                    if not isinstance(alert, dict):
                        continue
                    key = alert_dedup_key(alert)
                    severity = str(alert.get("severity", "info"))
                    rule_id = str(alert.get("rule_id", "—"))
                    call_id = alert.get("call_id", "—")
                    message = alert.get("message", alert.get("rule_id", "Alert"))
                    actions = alert.get("recommended_actions") or []

                    with ui.card().classes("w-full q-mb-sm"):
                        with ui.row().classes("w-full items-start justify-between gap-2"):
                            with ui.column().classes("flex-grow"):
                                with ui.row().classes("items-center gap-2 flex-wrap"):
                                    ui.chip(severity, color=severity_color(severity))
                                    ui.chip(f"Regel: {rule_id}", color="grey").props("dense")
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
                                    ).props("flat dense size=sm")
                                ui.button(
                                    icon="check",
                                    text="Markera hanterad",
                                    on_click=lambda k=key: _dismiss(k),
                                ).props("flat dense color=positive size=sm")

        # Handled alerts section
        handled_keys = state.dismissed_alert_keys
        if handled_keys:
            with ui.expansion(f"Hanterade alerts ({len(handled_keys)})", icon="done_all").classes("w-full q-mt-md"):
                ui.label("Alerts du markerat som hanterade i denna session.").classes("text-caption q-mb-sm")
                for key in handled_keys[-10:]:  # show last 10
                    ui.label(f"✓ {key.split('|')[-1][:60]}").classes("text-caption text-grey q-my-xs")

    def _export_active_alerts(alerts: list[dict[str, Any]]) -> None:
        payload = json.dumps(alerts, indent=2, ensure_ascii=False)
        filename = f"active_alerts_{datetime.now():%Y%m%d_%H%M%S}.json"
        ui.download(payload.encode("utf-8"), filename)
        ui.notify(f"Exporterade {len(alerts)} alerts", type="positive")

    def _dismiss(key: str) -> None:
        if key not in state.dismissed_alert_keys:
            state.dismissed_alert_keys.append(key)
            ui.notify("Alert markerad som hanterad", type="positive", close_button=True)
        alerts_section.refresh()
        if on_dismiss_change:
            on_dismiss_change()

    alerts_section()
    return alerts_section.refresh