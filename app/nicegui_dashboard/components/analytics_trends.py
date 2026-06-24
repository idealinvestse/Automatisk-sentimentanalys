"""Analytics & Trends tab – Plotly charts for sentiment, agents, topics.

Fas 6.1 – docs/MIGRATION_TO_NICEGUI_PLAN.md (Plotly/Echarts integration)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nicegui import ui

from app.nicegui_dashboard.components.call_detail import find_report
from app.nicegui_dashboard.components.emotion_timeline import render_emotion_timeline
from app.nicegui_dashboard.components.hot_topic_wordcloud import render_hot_topics_wordcloud
from app.nicegui_dashboard.services.chart_data import (
    build_agent_trends_figure,
    build_escalation_figure,
    build_hot_topics_figure,
    build_trajectory_figure,
    call_id_from_plotly_click,
    extract_agent_trend_rows,
    list_call_options,
)
from app.nicegui_dashboard.services.calls_filter import search_table_reports
from app.nicegui_dashboard.state import DashboardState
from app.services.data_services import filter_reports


def render_analytics_tab(
    state: DashboardState,
    *,
    on_call_select: Callable[[str], None] | None = None,
) -> Callable[[], None]:
    """Render analytics charts. Returns refresh callback."""

    if not state.filters.get("analytics_call_id"):
        if state.reports:
            state.filters["analytics_call_id"] = state.reports[0].get("call_id")

    def _filtered_reports() -> list[dict[str, Any]]:
        base = filter_reports(state.reports, state.filters)
        return search_table_reports(base, state.table_search)

    def _handle_plotly_click(e: Any) -> None:
        args = getattr(e, "args", e) or {}
        call_id = call_id_from_plotly_click(args if isinstance(args, dict) else {})
        if call_id and on_call_select:
            ui.notify(f"Öppnar {call_id}")
            on_call_select(call_id)

    @ui.refreshable
    def charts_section() -> None:
        reports = _filtered_reports()
        trend_rows = extract_agent_trend_rows(reports)
        options = list_call_options(reports)
        selected_id = state.filters.get("analytics_call_id")
        if options and selected_id not in {o["value"] for o in options}:
            selected_id = options[0]["value"]
            state.filters["analytics_call_id"] = selected_id

        report = find_report(reports, selected_id)

        ui.label("📈 Analys & Trender").classes("text-h6 q-mb-sm")
        ui.label(
            "Klicka på en punkt/stapel för att öppna relaterat samtal i Samtalsdetalj."
        ).classes("text-caption q-mb-md")

        with ui.row().classes("w-full gap-4 flex-wrap"):
            with ui.card().classes("flex-1 min-w-[320px]"):
                ui.label("Kundsentiment över tid").classes("text-subtitle2")
                if options:
                    ui.select(
                        options={o["value"]: o["label"] for o in options},
                        value=selected_id,
                        label="Välj samtal",
                        on_change=lambda e: (
                            state.filters.update({"analytics_call_id": e.value}),
                            charts_section.refresh(),
                        ),
                    ).classes("w-full q-mb-sm").props("dense")
                traj_plot = ui.plotly(build_trajectory_figure(report)).classes("w-full")
                traj_plot.on("plotly_click", _handle_plotly_click)

            with ui.card().classes("flex-1 min-w-[320px]"):
                ui.label("Agentprestanda över tid").classes("text-subtitle2")
                agent_plot = ui.plotly(build_agent_trends_figure(trend_rows)).classes("w-full")
                agent_plot.on("plotly_click", _handle_plotly_click)

        with ui.row().classes("w-full gap-4 flex-wrap q-mt-md"):
            with ui.card().classes("flex-1 min-w-[280px]"):
                ui.label("Heta ämnen").classes("text-subtitle2")
                topics_plot = ui.plotly(build_hot_topics_figure(reports)).classes("w-full")

            with ui.card().classes("flex-1 min-w-[280px]"):
                ui.label("Eskaleringstrender").classes("text-subtitle2")
                esc_plot = ui.plotly(build_escalation_figure(trend_rows)).classes("w-full")
                esc_plot.on("plotly_click", _handle_plotly_click)

        # Fas 3 viz additions (emotion timeline + hot topics wordcloud/treemap)
        with ui.row().classes("w-full gap-4 flex-wrap q-mt-md"):
            with ui.card().classes("flex-1 min-w-[420px]"):
                ui.label("Emotion timeline").classes("text-subtitle2")
                render_emotion_timeline(report)

            with ui.card().classes("flex-1 min-w-[320px]"):
                ui.label("Heta ämnen (treemap)").classes("text-subtitle2")
                render_hot_topics_wordcloud(reports)

    charts_section()
    return charts_section.refresh