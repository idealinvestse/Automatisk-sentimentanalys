"""Fas 4 Insikter tab – composes QA, insights, search and alerts sections.

Fas 3 – docs/archive/GROK_BUILD_PLAN_FAS1-3.md
"""

from __future__ import annotations

from collections.abc import Callable

from nicegui import ui

from app.nicegui_dashboard.components.alerts_panel import render_alerts_panel
from app.nicegui_dashboard.components.call_detail import find_report
from app.nicegui_dashboard.components.ui_primitives import metric_card, render_tab_header
from app.nicegui_dashboard.components.insights_hot_topics import render_insights_section
from app.nicegui_dashboard.components.pii_audit import render_pii_audit_panel
from app.nicegui_dashboard.components.qa_scorecard import render_qa_scorecard_section
from app.nicegui_dashboard.services.analytics_summary import total_pii_events
from app.nicegui_dashboard.state import DashboardState
from app.services.data_services import compute_kpis, filter_reports


def render_fas4_insights_tab(
    state: DashboardState,
    *,
    on_call_select: Callable[[str], None] | None = None,
    on_alerts_change: Callable[[], None] | None = None,
    on_topic_filter: Callable[[], None] | None = None,
) -> Callable[[], None]:
    """Assemble Fas 4 insights tab. Returns combined refresh callback."""
    refreshers: list[Callable[[], None]] = []

    with ui.column().classes("w-full q-gutter-y-md"):
        render_tab_header(
            "Fas 4 – Insikter",
            hint=(
                "QA-bedömning, hetämnen, semantisk sökning och aviseringar – "
                "klicka för att öppna samtal i detaljvyn."
            ),
        )

        @ui.refreshable
        def kpi_header() -> None:
            reports = filter_reports(state.reports, state.filters)
            kpis = compute_kpis(reports, state.filters)
            qa_display = (
                f"{kpis['qa_avg']}/100" if kpis.get("qa_avg") is not None else "—"
            )
            with ui.row().classes("w-full gap-4 flex-wrap q-mb-sm"):
                metric_card("Totalt", kpis.get("total_calls", 0), size="compact")
                metric_card("QA snitt", qa_display, color="warning", size="compact")
                metric_card(
                    "Alerts",
                    kpis.get("alerts_count", 0),
                    color="negative",
                    size="compact",
                )
                metric_card(
                    "Riskabla",
                    kpis.get("risky_calls", 0),
                    color="negative",
                    size="compact",
                )
            ui.badge(
                f"PII-händelser totalt: {total_pii_events(reports)}",
                color="orange",
            ).classes("q-mb-md")

        kpi_header()
        refreshers.append(kpi_header.refresh)

        refreshers.append(
            render_qa_scorecard_section(state, on_call_select=on_call_select)
        )

        def _on_topic_select(topic: str) -> None:
            state.filters["topic_filter"] = topic.lower()
            ui.notify(f"Filter på ämne: {topic}", type="info")
            if on_topic_filter:
                on_topic_filter()

        refreshers.append(
            render_insights_section(
                state,
                on_call_select=on_call_select,
                on_topic_select=_on_topic_select,
            )
        )

        refreshers.append(
            render_alerts_panel(
                state,
                on_call_select=on_call_select,
                on_dismiss_change=on_alerts_change,
            )
        )

        @ui.refreshable
        def pii_panel() -> None:
            cid = state.selected_qa_call_id or state.selected_call_id
            report = find_report(state.reports, cid) if cid else None
            render_pii_audit_panel(report.get("results") if report else None)

        pii_panel()
        refreshers.append(pii_panel.refresh)

    def refresh_all() -> None:
        for fn in refreshers:
            fn()

    return refresh_all