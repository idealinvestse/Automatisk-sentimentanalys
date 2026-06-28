"""Overview tab: KPIs, filters, hot topics, leaderboard, calls table.

Fas 3 – docs/archive/MIGRATION_TO_NICEGUI_PLAN.md §3
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nicegui import ui

from app.nicegui_dashboard.components.calls_table import render_calls_table
from app.nicegui_dashboard.components.empty_state import render_empty_state
from app.nicegui_dashboard.components.kpi_cards import render_kpi_row
from app.nicegui_dashboard.components.ui_primitives import render_section_title, render_tab_header
from app.nicegui_dashboard.settings import is_dev_mode
from app.nicegui_dashboard.services.analytics_summary import (
    build_calls_overview_rows,
    overview_csv_filename,
    overview_rows_to_csv_bytes,
)
from app.nicegui_dashboard.services.chart_data import build_sentiment_distribution_figure
from app.nicegui_dashboard.state import DashboardState
from app.services.data_services import filter_reports, get_agent_leaderboard, get_hot_topics

_MIN_RISK_OPTIONS = {
    "all": "Alla risknivåer",
    "low": "Låg",
    "medium": "Medel",
    "high": "Hög",
}


def _unique_agents(reports: list[dict[str, Any]]) -> list[str]:
    agents = sorted({(r.get("meta") or {}).get("agent", "Okänd") for r in reports})
    return ["Alla"] + agents


def _filters_active(state: DashboardState) -> bool:
    sentiment = state.filters.get("sentiment_filter", "all")
    agent = state.filters.get("agent_filter")
    has_qa_fail = state.filters.get("has_qa_fail")
    min_risk = state.filters.get("min_risk")
    return (
        sentiment != "all"
        or bool(agent)
        or has_qa_fail is True
        or bool(min_risk and min_risk != "all")
    )


def render_overview_tab(
    state: DashboardState,
    *,
    on_call_select: Callable[[str], None] | None = None,
    on_agent_select: Callable[[str], None] | None = None,
    on_show_example_detail: Callable[[], None] | None = None,
    on_reload_api: Callable[[], None] | None = None,
) -> Callable[[], None]:
    """Assemble the overview tab with reactive filters. Returns refresh callback."""

    @ui.refreshable
    def header_section() -> None:
        render_tab_header("Översikt – KPI:er & filter", meta=f"Data: {state.data_source}")

    header_section()

    if not state.reports:
        render_empty_state(
            icon="inbox",
            title="Ingen data laddad",
            hint="Starta backend-API eller ladda om för att hämta samtal.",
            action=on_reload_api,
            action_label="Ladda från API",
        )
        return header_section.refresh

    agents = _unique_agents(state.reports)

    def get_filtered() -> list[dict[str, Any]]:
        return filter_reports(state.reports, state.filters)

    def _export_overview_csv() -> None:
        rows = build_calls_overview_rows(get_filtered())
        if not rows:
            ui.notify("Ingen data att exportera", type="warning")
            return
        ui.download(overview_rows_to_csv_bytes(rows), overview_csv_filename())
        ui.notify(f"Exporterade {len(rows)} samtal till CSV")

    @ui.refreshable
    def kpi_section() -> None:
        render_kpi_row(get_filtered(), state.filters)

    @ui.refreshable
    def topics_and_board() -> None:
        reports = get_filtered()
        with ui.row().classes("w-full gap-4 flex-wrap"):
            with ui.card().classes("flex-1 min-w-card"):
                render_section_title("Heta ämnen", icon="local_fire_department")
                topics = get_hot_topics(reports)[:6]
                if topics:
                    ui.table(
                        columns=[
                            {"name": "topic", "label": "Ämne", "field": "topic"},
                            {"name": "volume", "label": "Volym", "field": "volume"},
                        ],
                        rows=topics,
                        row_key="topic",
                        pagination={"rowsPerPage": 0},
                    ).classes("w-full").props("dense flat")
                    ui.label("Se alla i Fas 4 Insikter").classes("text-caption text-grey q-mt-xs")
                else:
                    render_empty_state(
                        icon="topic",
                        title="Inga heta ämnen",
                        hint="Kör pipeline på fler samtal för att se ämnesmönster.",
                    )

            with ui.card().classes("flex-1 min-w-card"):
                with ui.expansion("Agenttopplista", icon="groups", value=False).classes(
                    "w-full"
                ):
                    board = get_agent_leaderboard(reports)
                    board_table = ui.table(
                        columns=[
                            {"name": "agent", "label": "Agent", "field": "agent"},
                            {"name": "calls", "label": "Samtal", "field": "calls"},
                            {"name": "avg_empathy", "label": "Empati", "field": "avg_empathy"},
                            {"name": "avg_qa", "label": "QA", "field": "avg_qa"},
                        ],
                        rows=[
                            {
                                "agent": row["agent"],
                                "calls": row["calls"],
                                "avg_empathy": (
                                    row["avg_empathy"] if row["avg_empathy"] is not None else "—"
                                ),
                                "avg_qa": row["avg_qa"] if row["avg_qa"] is not None else "—",
                            }
                            for row in board
                        ],
                        row_key="agent",
                    ).classes("w-full")

                    def _leaderboard_click(e: Any) -> None:
                        row = e.args[1] if len(e.args) > 1 else e.args[0]
                        name = row.get("agent") if isinstance(row, dict) else None
                        if name and on_agent_select:
                            on_agent_select(name)

                    board_table.on("rowClick", _leaderboard_click)

            with ui.card().classes("flex-1 min-w-card"):
                render_section_title("Sentimentfördelning", icon="pie_chart")
                ui.plotly(build_sentiment_distribution_figure(reports)).classes(
                    "w-full chart-container-compact"
                )

    def apply_filters() -> None:
        state.table_page = 1
        kpi_section.refresh()
        topics_and_board.refresh()
        refresh_calls_table()

    with ui.card().classes("w-full q-mb-md"):
        render_section_title("Filter", icon="filter_alt")
        with ui.row().classes("w-full gap-4 flex-wrap items-end"):
            ui.select(
                options=["all", "positiv", "negativ"],
                label="Sentiment",
                value=state.filters.get("sentiment_filter", "all"),
                on_change=lambda e: (
                    state.filters.update({"sentiment_filter": e.value}),
                    apply_filters(),
                ),
            ).classes("min-w-32")
            ui.select(
                options=agents,
                label="Agent",
                value=state.filters.get("agent_filter") or "Alla",
                on_change=lambda e: (
                    state.filters.update(
                        {"agent_filter": None if e.value == "Alla" else e.value}
                    ),
                    apply_filters(),
                ),
            ).classes("min-w-40")
            ui.select(
                options=_MIN_RISK_OPTIONS,
                label="Min. risk",
                value=state.filters.get("min_risk") or "all",
                on_change=lambda e: (
                    state.filters.update(
                        {"min_risk": None if e.value == "all" else e.value}
                    ),
                    apply_filters(),
                ),
            ).classes("min-w-40")
            ui.checkbox(
                "Endast QA-underkända",
                value=state.filters.get("has_qa_fail") is True,
                on_change=lambda e: (
                    state.filters.update(
                        {"has_qa_fail": True if e.value else None}
                    ),
                    apply_filters(),
                ),
            )
            ui.button(
                "Exportera CSV",
                icon="download",
                on_click=_export_overview_csv,
            ).props("outline dense")

    @ui.refreshable
    def filter_empty_hint() -> None:
        filtered = get_filtered()
        if filtered or not _filters_active(state):
            return
        render_empty_state(
            icon="filter_alt_off",
            title="Inga samtal matchar valda filter",
            hint="Ändra sentiment-, agent- eller riskfilter för att se fler samtal.",
        )

    kpi_section()
    ui.separator()
    topics_and_board()
    filter_empty_hint()

    refresh_calls_table = render_calls_table(
        state,
        reports=get_filtered(),
        on_select=on_call_select,
    )

    if on_show_example_detail and is_dev_mode():
        ui.button("Visa exempel Samtalsdetalj", on_click=on_show_example_detail).classes("q-mt-md")

    def refresh_all() -> None:
        header_section.refresh()
        kpi_section.refresh()
        topics_and_board.refresh()
        filter_empty_hint.refresh()
        refresh_calls_table()

    return refresh_all