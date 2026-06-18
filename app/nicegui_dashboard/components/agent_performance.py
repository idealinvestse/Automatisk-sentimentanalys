"""Agent Performance tab – metrics, trends, leaderboard, drill-down.

Fas 3 – docs/GROK_BUILD_PLAN_FAS1-3.md
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nicegui import ui

from app.nicegui_dashboard.services.chart_data import (
    build_agent_trends_figure,
    call_id_from_plotly_click,
    extract_agent_trend_rows,
)
from app.nicegui_dashboard.services.fas4_data import (
    agent_leaderboard_rows,
    fetch_agent_performance,
    list_agent_ids,
    local_agent_metrics,
    reports_for_agent,
)
from app.nicegui_dashboard.services.ui_helpers import notify_api_error
from app.nicegui_dashboard.state import DashboardState
from app.services.data_services import filter_reports


def _metric_card(label: str, value: Any, *, suffix: str = "") -> None:
    with ui.card().classes("flex-1 min-w-[140px]"):
        ui.label(label).classes("text-caption text-grey")
        display = "—" if value is None else f"{value}{suffix}"
        ui.label(str(display)).classes("text-h6")


def render_agent_performance_tab(
    state: DashboardState,
    *,
    on_call_select: Callable[[str], None] | None = None,
    on_agent_select: Callable[[str], None] | None = None,
) -> Callable[[], None]:
    """Render Agent Performance view. Returns refresh callback."""
    metrics_source = {"value": "local"}
    loading = {"active": False}
    metrics_holder: dict[str, Any] = {"data": {}}
    initial_api_load_done = {"value": False}

    def _filtered() -> list[dict[str, Any]]:
        return filter_reports(state.reports, state.filters)

    def _ensure_agent(reports: list[dict[str, Any]]) -> str | None:
        agents = list_agent_ids(reports)
        if not agents:
            return None
        selected = state.selected_agent_id
        if selected not in agents:
            selected = agents[0]
            state.selected_agent_id = selected
        return selected

    async def _load_metrics(agent_id: str, reports: list[dict[str, Any]]) -> dict[str, Any]:
        if state.api_client and state.api_connected:
            metrics, source = await fetch_agent_performance(state.api_client, agent_id, reports)
            metrics_source["value"] = source
            return metrics
        metrics_source["value"] = "local"
        return local_agent_metrics(agent_id, reports)

    async def _refresh_metrics_for_current() -> None:
        reports = _filtered()
        agent_id = _ensure_agent(reports)
        if not agent_id or loading["active"]:
            return
        loading["active"] = True
        try:
            metrics_holder["data"] = await _load_metrics(agent_id, reports)
        except Exception as err:
            notify_api_error(err)
            metrics_holder["data"] = local_agent_metrics(agent_id, reports)
            metrics_source["value"] = "local"
        finally:
            loading["active"] = False
        performance_section.refresh()

    def _handle_plotly_click(e: Any) -> None:
        args = getattr(e, "args", e) or {}
        call_id = call_id_from_plotly_click(args if isinstance(args, dict) else {})
        if call_id and on_call_select:
            on_call_select(call_id)

    @ui.refreshable
    def performance_section() -> None:
        reports = _filtered()
        agent_id = _ensure_agent(reports)

        with ui.row().classes("w-full items-center justify-between q-mb-md"):
            ui.label("👤 Agent Performance").classes("text-h6")
            ui.label(f"Metrics: {metrics_source['value']}").classes("text-caption")

        if not reports or not agent_id:
            ui.label("Ingen agentdata tillgänglig.").classes("text-caption")
            return

        agents = list_agent_ids(reports)
        agent_reports = reports_for_agent(reports, agent_id)
        trend_rows = [r for r in extract_agent_trend_rows(agent_reports) if r.get("agent") == agent_id]
        metrics = metrics_holder["data"] or local_agent_metrics(agent_id, reports)

        async def _on_agent_change(e: Any) -> None:
            state.selected_agent_id = e.value
            metrics_holder["data"] = local_agent_metrics(e.value, reports)
            if on_agent_select:
                on_agent_select(e.value)
            performance_section.refresh()
            await _refresh_metrics_for_current()

        with ui.row().classes("w-full gap-4 flex-wrap items-end q-mb-md"):
            ui.select(
                options=agents,
                value=agent_id,
                label="Välj agent",
                on_change=_on_agent_change,
            ).classes("min-w-48")
            ui.button(
                icon="refresh",
                on_click=_refresh_metrics_for_current,
            ).props("flat round dense").tooltip("Uppdatera agent-metrics")

        avgs = metrics.get("averages") or {}
        with ui.row().classes("w-full gap-3 flex-wrap"):
            _metric_card("Samtal", metrics.get("call_count", 0))
            _metric_card("Empati", avgs.get("empathy_score"))
            _metric_card("Talk ratio", avgs.get("talk_ratio"))
            _metric_card("De-eskalering", avgs.get("de_escalation_effectiveness"))
            with ui.card().classes("flex-1 min-w-[140px]"):
                ui.label("Trend empati").classes("text-caption text-grey")
                ui.label(str(metrics.get("trend_empathy", "—"))).classes("text-h6")

        flags = metrics.get("compliance_flags") or []
        if flags:
            ui.label("Compliance-flaggor").classes("text-subtitle2 q-mt-sm")
            with ui.row().classes("flex-wrap"):
                for flag in flags[:8]:
                    ui.chip(str(flag), color="warning").classes("q-ma-xs")

        with ui.card().classes("w-full q-mt-md"):
            ui.label("Trend per samtal").classes("text-subtitle2")
            plot = ui.plotly(build_agent_trends_figure(trend_rows)).classes("w-full")
            plot.on("plotly_click", _handle_plotly_click)

        with ui.card().classes("w-full q-mt-md"):
            ui.label("Leaderboard").classes("text-subtitle2")
            board = agent_leaderboard_rows(reports)
            board_table = ui.table(
                columns=[
                    {"name": "agent", "label": "Agent", "field": "agent"},
                    {"name": "calls", "label": "Samtal", "field": "calls"},
                    {"name": "avg_empathy", "label": "Empati", "field": "avg_empathy"},
                    {"name": "avg_qa", "label": "QA", "field": "avg_qa"},
                ],
                rows=board,
                row_key="agent",
            ).classes("w-full")

            async def _leaderboard_click(e: Any) -> None:
                row = e.args[1] if len(e.args) > 1 else e.args[0]
                name = row.get("agent") if isinstance(row, dict) else None
                if name and name in agents:
                    state.selected_agent_id = name
                    metrics_holder["data"] = local_agent_metrics(name, reports)
                    performance_section.refresh()
                    await _refresh_metrics_for_current()

            board_table.on("rowClick", _leaderboard_click)

        with ui.card().classes("w-full q-mt-md"):
            ui.label("Samtal för vald agent").classes("text-subtitle2")
            rows = [
                {
                    "call_id": r.get("call_id"),
                    "title": r.get("title", ""),
                    "agent": (r.get("meta") or {}).get("agent", ""),
                }
                for r in agent_reports
            ]
            calls_table = ui.table(
                columns=[
                    {"name": "call_id", "label": "ID", "field": "call_id"},
                    {"name": "title", "label": "Titel", "field": "title"},
                ],
                rows=rows,
                row_key="call_id",
            ).classes("w-full")

            def _call_row_click(e: Any) -> None:
                if not on_call_select:
                    return
                row = e.args[1] if len(e.args) > 1 else e.args[0]
                cid = row.get("call_id") if isinstance(row, dict) else None
                if cid:
                    on_call_select(str(cid))

            calls_table.on("rowClick", _call_row_click)

    async def _initial_api_load() -> None:
        if initial_api_load_done["value"]:
            return
        initial_api_load_done["value"] = True
        if state.api_client and state.api_connected:
            await _refresh_metrics_for_current()

    performance_section()
    ui.timer(0.1, _initial_api_load, once=True)
    return performance_section.refresh