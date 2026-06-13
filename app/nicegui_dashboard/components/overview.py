"""Overview tab: KPIs, filters, hot topics, leaderboard, calls table.

Fas 3 – docs/MIGRATION_TO_NICEGUI_PLAN.md §3
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nicegui import ui

from app.nicegui_dashboard.components.calls_table import render_calls_table
from app.nicegui_dashboard.components.kpi_cards import render_kpi_row
from app.nicegui_dashboard.state import DashboardState
from app.services.data_services import filter_reports, get_agent_leaderboard, get_hot_topics


def _unique_agents(reports: list[dict[str, Any]]) -> list[str]:
    agents = sorted({(r.get("meta") or {}).get("agent", "Okänd") for r in reports})
    return ["Alla"] + agents


def render_overview_tab(
    state: DashboardState,
    *,
    on_call_select: Callable[[str], None] | None = None,
    on_show_example_detail: Callable[[], None] | None = None,
    on_reload_api: Callable[[], None] | None = None,
) -> Callable[[], None]:
    """Assemble the overview tab with reactive filters. Returns refresh callback."""
    with ui.row().classes("w-full items-center justify-between q-mb-md"):
        ui.label("📊 Översikt – KPI:er & Filter").classes("text-h6")
        source_label = ui.label(f"Data: {state.data_source}").classes("text-caption")
        if state.api_client and on_reload_api:
            ui.button("↻ Ladda från API", on_click=on_reload_api).props("flat dense")

    agents = _unique_agents(state.reports)

    def get_filtered() -> list[dict[str, Any]]:
        return filter_reports(state.reports, state.filters)

    @ui.refreshable
    def kpi_section() -> None:
        render_kpi_row(get_filtered(), state.filters)

    @ui.refreshable
    def topics_and_board() -> None:
        reports = get_filtered()
        with ui.row().classes("w-full gap-4"):
            with ui.card().classes("flex-1"):
                ui.label("🔥 Hot Topics").classes("text-subtitle2")
                topics = get_hot_topics(reports)
                if topics:
                    for item in topics:
                        ui.chip(item.get("topic", "okänt"), color="primary").classes("q-ma-xs")
                else:
                    for topic in ["Fakturering", "Teknisk support", "Väntetid", "Empati"]:
                        ui.chip(topic, color="primary").classes("q-ma-xs")

            with ui.card().classes("flex-1"):
                ui.label("👥 Agent Leaderboard").classes("text-subtitle2")
                board = get_agent_leaderboard(reports)
                ui.table(
                    columns=[
                        {"name": "agent", "label": "Agent", "field": "agent"},
                        {"name": "calls", "label": "Samtal", "field": "calls"},
                        {"name": "avg_qa", "label": "QA", "field": "avg_qa"},
                    ],
                    rows=[
                        {
                            "agent": row["agent"],
                            "calls": row["calls"],
                            "avg_qa": row["avg_qa"] if row["avg_qa"] is not None else "—",
                        }
                        for row in board
                    ],
                    row_key="agent",
                ).classes("w-full")

    def apply_filters() -> None:
        kpi_section.refresh()
        topics_and_board.refresh()
        calls_section.refresh()

    with ui.card().classes("w-full q-mb-md"):
        ui.label("Filter").classes("text-subtitle2")
        with ui.row().classes("w-full gap-4 flex-wrap"):
            ui.select(
                "Sentiment",
                options=["all", "positiv", "negativ"],
                value=state.filters.get("sentiment_filter", "all"),
                on_change=lambda e: (
                    state.filters.update({"sentiment_filter": e.value}),
                    apply_filters(),
                ),
            ).classes("min-w-32")
            ui.select(
                "Agent",
                options=agents,
                value=state.filters.get("agent_filter") or "Alla",
                on_change=lambda e: (
                    state.filters.update(
                        {"agent_filter": None if e.value == "Alla" else e.value}
                    ),
                    apply_filters(),
                ),
            ).classes("min-w-40")
            ui.input(
                "Sök",
                value=state.filters.get("search", ""),
                on_change=lambda e: (
                    state.filters.update({"search": e.value or ""}),
                    apply_filters(),
                ),
            ).classes("flex-grow")

    kpi_section()
    ui.separator()
    topics_and_board()

    @ui.refreshable
    def calls_section() -> None:
        render_calls_table(state, reports=get_filtered(), on_select=on_call_select)

    calls_section()

    if on_show_example_detail:
        ui.button("Visa exempel Call Detail", on_click=on_show_example_detail).classes("q-mt-md")

    def refresh_all() -> None:
        source_label.set_text(f"Data: {state.data_source}")
        kpi_section.refresh()
        topics_and_board.refresh()
        calls_section.refresh()

    return refresh_all