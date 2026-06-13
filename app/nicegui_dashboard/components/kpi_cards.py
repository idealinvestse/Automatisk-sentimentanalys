"""KPI metric cards for the overview tab.

Fas 1 – docs/MIGRATION_TO_NICEGUI_PLAN.md §3 (Portning av KPI-metrics)
"""

from __future__ import annotations

from typing import Any

from nicegui import ui

from app.services.data_services import compute_kpis


def render_kpi_row(reports: list[dict[str, Any]], filters: dict[str, Any] | None = None) -> None:
    """Render KPI cards computed from reports via data_services.compute_kpis."""
    kpis = compute_kpis(reports, filters)
    qa_display = f"{kpis['qa_avg']}/100" if kpis.get("qa_avg") is not None else "—"

    cards = [
        ("Totalt samtal", str(kpis.get("total_calls", 0)), "primary"),
        ("Positiva", f"{kpis.get('pos_pct', 0)}%", "positive"),
        ("Negativa", f"{kpis.get('neg_pct', 0)}%", "negative"),
        ("QA Snitt", qa_display, "warning"),
        ("Alerts", str(kpis.get("alerts_count", 0)), "negative"),
    ]

    with ui.row().classes("w-full gap-4"):
        for label, value, color in cards:
            with ui.card().classes("flex-1"):
                ui.label(label).classes("text-caption")
                ui.label(value).classes(f"text-h4 text-{color}")