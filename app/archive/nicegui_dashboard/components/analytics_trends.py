"""Analytics & Trends tab – Plotly charts for sentiment, agents, topics.

Fas 6.1 – docs/archive/MIGRATION_TO_NICEGUI_PLAN.md (Plotly/Echarts integration)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nicegui import ui

from app.archive.nicegui_dashboard.components.call_detail import find_report
from app.archive.nicegui_dashboard.components.emotion_timeline import render_emotion_timeline
from app.archive.nicegui_dashboard.components.empty_state import render_empty_state
from app.archive.nicegui_dashboard.components.hot_topic_wordcloud import render_hot_topics_wordcloud
from app.archive.nicegui_dashboard.components.ui_primitives import (
    metric_card,
    render_section_title,
    render_tab_header,
)
from app.archive.nicegui_dashboard.services.analytics_summary import (
    build_calls_overview_rows,
    compute_call_snapshot,
    compute_portfolio_kpis,
    filter_reports_by_agent,
    list_agent_options,
    overview_csv_filename,
    overview_rows_to_csv_bytes,
    summarize_emotions,
)
from app.archive.nicegui_dashboard.services.calls_filter import search_table_reports
from app.archive.nicegui_dashboard.services.chart_data import (
    build_agent_trends_figure,
    build_escalation_figure,
    build_hot_topics_figure,
    build_sentiment_distribution_figure,
    build_trajectory_figure,
    call_id_from_plotly_click,
    extract_agent_trend_rows,
    list_call_options,
)
from app.archive.nicegui_dashboard.services.qa_display import qa_chip_color
from app.archive.nicegui_dashboard.state import DashboardState
from app.services.data_services import filter_reports


def _format_duration(seconds: int | float | None) -> str:
    if not seconds:
        return "—"
    mins = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{mins} min {secs}s" if mins else f"{secs}s"


def render_analytics_tab(
    state: DashboardState,
    *,
    on_call_select: Callable[[str], None] | None = None,
) -> Callable[[], None]:
    """Render analytics charts. Returns refresh callback."""

    if not state.filters.get("analytics_call_id") and state.reports:
        state.filters["analytics_call_id"] = state.reports[0].get("call_id")
    if "analytics_agent_filter" not in state.filters:
        state.filters["analytics_agent_filter"] = "Alla"

    def _base_reports() -> list[dict[str, Any]]:
        return filter_reports(state.reports, state.filters)

    def _filtered_reports() -> list[dict[str, Any]]:
        base = _base_reports()
        by_agent = filter_reports_by_agent(base, state.filters.get("analytics_agent_filter"))
        return search_table_reports(by_agent, state.table_search)

    def _handle_plotly_click(e: Any) -> None:
        args = getattr(e, "args", e) or {}
        call_id = call_id_from_plotly_click(args if isinstance(args, dict) else {})
        if call_id and on_call_select:
            ui.notify(f"Öppnar {call_id}")
            on_call_select(call_id)

    def _select_call(call_id: str) -> None:
        state.filters["analytics_call_id"] = call_id
        state.selected_call_id = call_id
        charts_section.refresh()
        if on_call_select:
            on_call_select(call_id)

    def _export_overview_csv(rows: list[dict[str, Any]]) -> None:
        if not rows:
            ui.notify("Ingen data att exportera", type="warning")
            return
        ui.download(overview_rows_to_csv_bytes(rows), overview_csv_filename())
        ui.notify(f"Exporterade {len(rows)} samtal till CSV")

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
        snapshot = compute_call_snapshot(report)
        kpis = compute_portfolio_kpis(reports, state.filters)
        overview_rows = build_calls_overview_rows(reports)

        render_tab_header(
            "Analys & trender",
            hint="Utforska sentiment, agentprestanda och ämnen. Klicka i diagram eller tabell för samtalsdetalj.",
            meta=f"{len(reports)} samtal i urval",
        )

        if not reports:
            render_empty_state(
                icon="analytics",
                title="Ingen data för analys",
                hint="Justera filter under Översikt eller ladda fler samtal från API.",
            )
            return

        # --- Portfolio KPIs ---
        with ui.row().classes("w-full gap-3 flex-wrap q-mb-md"):
            metric_card("Samtal", kpis.get("total_calls", 0), size="compact")
            metric_card("Positiva", f"{kpis.get('pos_pct', 0)}%", color="positive", size="compact")
            metric_card("Negativa", f"{kpis.get('neg_pct', 0)}%", color="negative", size="compact")
            qa_val = f"{kpis['qa_avg']}/100" if kpis.get("qa_avg") is not None else "—"
            metric_card("QA-snitt", qa_val, color="warning", size="compact")
            metric_card(
                "Aviseringar", kpis.get("alerts_count", 0), color="negative", size="compact"
            )
            empathy_val = kpis.get("avg_empathy")
            metric_card(
                "Snitt empati",
                f"{empathy_val:.2f}" if empathy_val is not None else "—",
                color="info",
                size="compact",
            )
            metric_card("Agenter", kpis.get("unique_agents", 0), size="compact")

        # --- Filters ---
        with ui.card().classes("w-full q-mb-md"):
            render_section_title("Urval & fokus", icon="tune")
            with ui.row().classes("w-full gap-4 flex-wrap items-end"):
                agents = list_agent_options(_base_reports())
                ui.select(
                    options=agents,
                    value=state.filters.get("analytics_agent_filter", "Alla"),
                    label="Filtrera agent",
                    on_change=lambda e: (
                        state.filters.update({"analytics_agent_filter": e.value}),
                        charts_section.refresh(),
                    ),
                ).classes("min-w-48").props("dense")
                if options:
                    ui.select(
                        options={o["value"]: o["label"] for o in options},
                        value=selected_id,
                        label="Fokusera samtal",
                        on_change=lambda e: (
                            state.filters.update({"analytics_call_id": e.value}),
                            charts_section.refresh(),
                        ),
                    ).classes("flex-grow min-w-64").props("dense")
            ui.label(
                f"Toppämne: {kpis.get('top_topic', '—')} · "
                f"Riskfyllda samtal: {kpis.get('risky_calls', 0)}"
            ).classes("text-caption text-grey q-mt-xs")

        # --- Selected call context ---
        if snapshot:
            with ui.card().classes("w-full q-mb-md"):
                render_section_title(f"Valt samtal – {snapshot['call_id']}", icon="call")
                ui.label(snapshot.get("title", "")).classes("text-caption text-grey q-mb-sm")
                with ui.row().classes("gap-2 flex-wrap"):
                    ui.chip(f"Agent: {snapshot['agent']}", color="primary")
                    ui.chip(f"Längd: {_format_duration(snapshot.get('duration_s'))}")
                    ui.chip(f"Segment: {snapshot.get('segment_count', 0)}")
                    ui.chip(f"Kategori: {snapshot.get('category', '—')}")
                    sent = snapshot.get("sentiment_label", "neutral")
                    ui.chip(f"Sentiment: {sent}", color="secondary")
                    qa_score = snapshot.get("qa_score")
                    if qa_score is not None:
                        ui.chip(f"QA: {qa_score}/100", color=qa_chip_color(qa_score))
                    ui.chip(f"Aviseringar: {snapshot.get('alert_count', 0)}")
                    ui.chip(f"Trend: {snapshot.get('trajectory_trend', '—')}", color="info")
                if snapshot.get("trajectory_min") is not None:
                    ui.label(
                        f"Sentimentintervall: {snapshot['trajectory_min']} … {snapshot['trajectory_max']} · "
                        f"Negativa toppar: {snapshot.get('negative_peaks', 0)}"
                    ).classes("text-caption text-grey q-mt-xs")
                emotions = summarize_emotions(report)
                if emotions:
                    with ui.row().classes("gap-1 flex-wrap q-mt-sm"):
                        ui.label("Dominerande känslor:").classes("text-caption text-grey")
                        for emo in emotions[:4]:
                            ui.chip(
                                f"{emo['label_sv']} {emo['avg']:.2f}",
                                color="accent",
                            ).classes("text-caption")

        # --- Distribution + trends row ---
        with ui.row().classes("w-full gap-4 flex-wrap"):
            with ui.card().classes("flex-1 min-w-card"):
                render_section_title("Sentimentfördelning", icon="pie_chart")
                ui.plotly(build_sentiment_distribution_figure(reports)).classes(
                    "w-full chart-container"
                )

            with ui.card().classes("flex-1 min-w-card"):
                render_section_title("Kundsentiment över tid", icon="show_chart")
                ui.label(
                    "Per segment med glidande medel – negativa toppar markerar eskaleringsrisk."
                ).classes("text-caption text-grey q-mb-xs")
                traj_plot = ui.plotly(build_trajectory_figure(report)).classes(
                    "w-full chart-container"
                )
                traj_plot.on("plotly_click", _handle_plotly_click)

            with ui.card().classes("flex-1 min-w-card"):
                render_section_title("Agentprestanda över tid", icon="groups")
                ui.label("Empati och QA per samtal – streckad linje visar snittempati.").classes(
                    "text-caption text-grey q-mb-xs"
                )
                agent_plot = ui.plotly(build_agent_trends_figure(trend_rows)).classes(
                    "w-full chart-container"
                )
                agent_plot.on("plotly_click", _handle_plotly_click)

        with ui.row().classes("w-full gap-4 flex-wrap q-mt-md"):
            with ui.card().classes("flex-1 min-w-card"):
                render_section_title("Heta ämnen", icon="local_fire_department")
                ui.plotly(build_hot_topics_figure(reports)).classes("w-full chart-container")

            with ui.card().classes("flex-1 min-w-card"):
                render_section_title("Eskaleringstrender", icon="warning")
                ui.label("Antal aviseringar och risk per samtal.").classes(
                    "text-caption text-grey q-mb-xs"
                )
                esc_plot = ui.plotly(build_escalation_figure(trend_rows)).classes(
                    "w-full chart-container"
                )
                esc_plot.on("plotly_click", _handle_plotly_click)

        with ui.row().classes("w-full gap-4 flex-wrap q-mt-md"):
            with ui.card().classes("flex-1 min-w-card"):
                render_section_title("Emotionstidslinje", icon="timeline")
                render_emotion_timeline(report)

            with ui.card().classes("flex-1 min-w-card"):
                render_section_title("Heta ämnen (översikt)", icon="bubble_chart")
                render_hot_topics_wordcloud(reports)

        # --- Data table ---
        with ui.card().classes("w-full q-mt-md"):
            with ui.row().classes("w-full items-center justify-between q-mb-sm"):
                render_section_title("Detaljerad samtalsöversikt", icon="table_chart")
                ui.button(
                    "Exportera CSV",
                    icon="download",
                    on_click=lambda: _export_overview_csv(overview_rows),
                ).props("outline dense")

            ui.label("Klicka på en rad för att fokusera samtalet och öppna detaljvyn.").classes(
                "text-caption text-grey q-mb-sm"
            )
            table = ui.table(
                columns=[
                    {"name": "call_id", "label": "ID", "field": "call_id", "align": "left"},
                    {"name": "title", "label": "Ämne", "field": "title", "align": "left"},
                    {"name": "agent", "label": "Agent", "field": "agent", "align": "left"},
                    {
                        "name": "sentiment",
                        "label": "Sentiment",
                        "field": "sentiment",
                        "align": "left",
                    },
                    {"name": "empathy", "label": "Empati", "field": "empathy", "align": "left"},
                    {"name": "qa", "label": "QA", "field": "qa", "align": "left"},
                    {
                        "name": "escalation",
                        "label": "Aviseringar",
                        "field": "escalation",
                        "align": "left",
                    },
                    {"name": "segments", "label": "Segment", "field": "segments", "align": "left"},
                    {"name": "trend", "label": "Trend", "field": "trend", "align": "left"},
                ],
                rows=[
                    {
                        **row,
                        "empathy": row["empathy"] if row["empathy"] is not None else "—",
                        "qa": row["qa"] if row["qa"] is not None else "—",
                    }
                    for row in overview_rows
                ],
                row_key="call_id",
                pagination={"rowsPerPage": 10, "sortBy": "call_id"},
            ).classes("w-full")

            def _table_row_click(e: Any) -> None:
                row = e.args[1] if len(e.args) > 1 else e.args[0]
                cid = row.get("call_id") if isinstance(row, dict) else None
                if cid:
                    _select_call(str(cid))

            table.on("rowClick", _table_row_click)

    charts_section()
    return charts_section.refresh
