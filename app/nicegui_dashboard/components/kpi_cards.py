"""KPI metric cards for the overview tab.

Fas 1 – docs/archive/MIGRATION_TO_NICEGUI_PLAN.md §3 (Portning av KPI-metrics)
"""

from __future__ import annotations

from typing import Any

from nicegui import ui

from app.nicegui_dashboard.components.ui_primitives import metric_card
from app.nicegui_dashboard.services.analytics_summary import compute_portfolio_kpis


def render_kpi_row(reports: list[dict[str, Any]], filters: dict[str, Any] | None = None) -> None:
    """Render KPI cards computed from reports via analytics_summary.compute_portfolio_kpis."""
    kpis = compute_portfolio_kpis(reports, filters)
    qa_display = f"{kpis['qa_avg']}/100" if kpis.get("qa_avg") is not None else "—"

    cards = [
        ("Totalt samtal", str(kpis.get("total_calls", 0)), "primary"),
        ("Positiva", f"{kpis.get('pos_pct', 0)}%", "positive"),
        ("Negativa", f"{kpis.get('neg_pct', 0)}%", "negative"),
        ("QA Snitt", qa_display, "warning"),
        ("Alerts", str(kpis.get("alerts_count", 0)), "negative"),
        ("Riskfyllda samtal", str(kpis.get("risky_calls", 0)), "negative"),
        ("Heta ämnen", str(kpis.get("hot_topics_count", 0)), "info"),
        ("Agenter", str(kpis.get("unique_agents", 0)), "primary"),
    ]

    with ui.row().classes("w-full gap-3 flex-wrap"):
        for label, value, color in cards:
            metric_card(label, value, color=color, size="compact")
