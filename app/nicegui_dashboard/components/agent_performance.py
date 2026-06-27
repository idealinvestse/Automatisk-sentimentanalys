"""Agentprestanda-fliken – metrics, trends, leaderboard, drill-down.

Fas 3 – docs/GROK_BUILD_PLAN_FAS1-3.md
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nicegui import ui

from app.nicegui_dashboard.components.empty_state import render_empty_state
from app.nicegui_dashboard.components.ui_primitives import metric_card, render_section_title, render_tab_header
from app.nicegui_dashboard.services.analytics_summary import (
    aggregate_agent_stats,
    build_calls_overview_rows,
)
from app.nicegui_dashboard.services.chart_data import (
    build_agent_trends_figure,
    build_escalation_figure,
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


def _metric_value(value: Any) -> str | int | float:
    return "—" if value is None else value


def _collect_coaching_recommendations(agent_reports: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Gather coaching items from agent_assessment across an agent's calls."""
    recs: list[dict[str, str]] = []
    for report in agent_reports:
        call_id = str(report.get("call_id") or report.get("id", "?"))
        assess = (report.get("results") or {}).get("agent_assessment") or (
            (report.get("llm") or {}).get("agent_assessment")
        ) or {}
        if not isinstance(assess, dict):
            continue
        for item in assess.get("specific_coaching_recommendations") or []:
            if isinstance(item, dict):
                text = (
                    item.get("recommendation")
                    or item.get("text")
                    or item.get("title")
                    or str(item)
                )
            else:
                text = str(item)
            if text.strip():
                recs.append({"call_id": call_id, "text": text.strip()})
    return recs


def _aggregate_customer_metrics(agent_reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Average customer-side metrics when present in report results."""
    talk_vals: list[float] = []
    resolution_vals: list[float] = []
    frustration_peaks = 0
    question_count = 0
    for report in agent_reports:
        cm = (report.get("results") or {}).get("customer_metrics") or {}
        if not isinstance(cm, dict):
            continue
        if cm.get("talk_ratio") is not None:
            talk_vals.append(float(cm["talk_ratio"]))
        frustration_peaks += int(cm.get("frustration_peaks", 0) or 0)
        question_count += int(cm.get("question_count", 0) or 0)
        if cm.get("resolution_indicators") is not None:
            resolution_vals.append(float(cm["resolution_indicators"]))
    return {
        "talk_ratio": round(sum(talk_vals) / len(talk_vals), 2) if talk_vals else None,
        "frustration_peaks": frustration_peaks if frustration_peaks else None,
        "question_count": question_count if question_count else None,
        "resolution_indicators": (
            round(sum(resolution_vals) / len(resolution_vals), 2) if resolution_vals else None
        ),
    }


def render_agent_performance_tab(
    state: DashboardState,
    *,
    on_call_select: Callable[[str], None] | None = None,
    on_agent_select: Callable[[str], None] | None = None,
) -> Callable[[], None]:
    """Render Agentprestanda view. Returns refresh callback."""
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

        render_tab_header(
            "Agentprestanda",
            meta=f"Källor: {metrics_source['value']}",
        )

        if not reports or not agent_id:
            render_empty_state(
                icon="person_off",
                title="Ingen agentdata tillgänglig",
                hint="Ladda samtal eller justera filter för att se agentprestanda.",
            )
            return

        agents = list_agent_ids(reports)
        agent_reports = reports_for_agent(reports, agent_id)
        trend_rows = [r for r in extract_agent_trend_rows(agent_reports) if r.get("agent") == agent_id]
        metrics = metrics_holder["data"] or local_agent_metrics(agent_id, reports)
        agent_stats = aggregate_agent_stats(reports, agent_id)
        customer_metrics = _aggregate_customer_metrics(agent_reports)
        coaching_recs = _collect_coaching_recommendations(agent_reports)
        overview_rows = build_calls_overview_rows(agent_reports)

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

        with ui.card().classes("w-full q-mb-md"):
            render_section_title(f"Sammanfattning – {agent_id}", icon="insights")
            with ui.row().classes("w-full gap-3 flex-wrap"):
                metric_card("Samtal", agent_stats.get("call_count", 0), size="compact")
                metric_card("Snitt empati", _metric_value(agent_stats.get("avg_empathy")), size="compact")
                qa_avg = agent_stats.get("avg_qa")
                metric_card(
                    "Snitt QA",
                    f"{qa_avg}/100" if qa_avg is not None else "—",
                    size="compact",
                    color="warning",
                )
                metric_card("Aviseringar", agent_stats.get("alert_count", 0), size="compact", color="negative")

        avgs = metrics.get("averages") or {}
        with ui.row().classes("w-full gap-3 flex-wrap"):
            metric_card("Samtal", metrics.get("call_count", 0), size="compact")
            metric_card("Empati", _metric_value(avgs.get("empathy_score")), size="compact")
            metric_card("Taltid-andel", _metric_value(avgs.get("talk_ratio")), size="compact")
            metric_card(
                "De-eskalering",
                _metric_value(avgs.get("de_escalation_effectiveness")),
                size="compact",
            )
            metric_card(
                "Trend empati",
                _metric_value(metrics.get("trend_empathy")),
                size="compact",
            )

        if any(customer_metrics.get(k) is not None for k in customer_metrics):
            ui.label("Kundsignaler").classes("text-subtitle2 q-mt-sm")
            with ui.row().classes("flex-wrap gap-1"):
                if customer_metrics.get("talk_ratio") is not None:
                    ui.chip(
                        f"Taltid kund: {customer_metrics['talk_ratio']:.0%}",
                        color="primary",
                    ).classes("q-ma-xs")
                if customer_metrics.get("frustration_peaks") is not None:
                    ui.chip(
                        f"Frustrationstoppar: {customer_metrics['frustration_peaks']}",
                        color="negative",
                    ).classes("q-ma-xs")
                if customer_metrics.get("question_count") is not None:
                    ui.chip(
                        f"Frågor: {customer_metrics['question_count']}",
                        color="info",
                    ).classes("q-ma-xs")
                if customer_metrics.get("resolution_indicators") is not None:
                    ui.chip(
                        f"Lösningsindikator: {customer_metrics['resolution_indicators']:.2f}",
                        color="positive",
                    ).classes("q-ma-xs")

        flags = metrics.get("compliance_flags") or []
        if flags:
            ui.label("Compliance-flaggor").classes("text-subtitle2 q-mt-sm")
            with ui.row().classes("flex-wrap"):
                for flag in flags[:8]:
                    ui.chip(str(flag), color="warning").classes("q-ma-xs")

        with ui.expansion("Coaching-rekommendationer", icon="school", value=False).classes(
            "w-full q-mt-md"
        ):
            if coaching_recs:
                for rec in coaching_recs[:12]:
                    ui.label(f"{rec['call_id']}: {rec['text']}").classes("text-body2 q-mb-xs")
            else:
                ui.label("Inga coaching-rekommendationer för vald agent.").classes(
                    "text-caption text-grey"
                )

        with ui.row().classes("w-full gap-4 flex-wrap q-mt-md"):
            with ui.card().classes("flex-1 min-w-card"):
                ui.label("Trend per samtal").classes("text-subtitle2")
                plot = ui.plotly(build_agent_trends_figure(trend_rows)).classes("w-full")
                plot.on("plotly_click", _handle_plotly_click)

            with ui.card().classes("flex-1 min-w-card"):
                ui.label("Eskaleringstrender").classes("text-subtitle2")
                esc_plot = ui.plotly(build_escalation_figure(trend_rows)).classes("w-full")
                esc_plot.on("plotly_click", _handle_plotly_click)

        with ui.card().classes("w-full q-mt-md"):
            ui.label("Agenttopplista").classes("text-subtitle2")
            board = agent_leaderboard_rows(reports)
            board_table = ui.table(
                columns=[
                    {"name": "agent", "label": "Agent", "field": "agent"},
                    {"name": "calls", "label": "Samtal", "field": "calls"},
                    {"name": "avg_empathy", "label": "Empati", "field": "avg_empathy"},
                    {"name": "avg_qa", "label": "QA", "field": "avg_qa"},
                    {"name": "coaching_recs", "label": "Coaching", "field": "coaching_recs"},
                ],
                rows=[
                    {
                        **row,
                        "avg_empathy": row["avg_empathy"] if row.get("avg_empathy") is not None else "—",
                        "avg_qa": row["avg_qa"] if row.get("avg_qa") is not None else "—",
                    }
                    for row in board
                ],
                row_key="agent",
            ).classes("w-full")

            async def _leaderboard_click(e: Any) -> None:
                row = e.args[1] if len(e.args) > 1 else e.args[0]
                name = row.get("agent") if isinstance(row, dict) else None
                if name and name in agents:
                    state.selected_agent_id = name
                    metrics_holder["data"] = local_agent_metrics(name, reports)
                    if on_agent_select:
                        on_agent_select(name)
                    performance_section.refresh()
                    await _refresh_metrics_for_current()

            board_table.on("rowClick", _leaderboard_click)

        with ui.card().classes("w-full q-mt-md"):
            ui.label("Samtal för vald agent").classes("text-subtitle2")
            rows = [
                {
                    "call_id": row["call_id"],
                    "title": row.get("title", ""),
                    "sentiment": row.get("sentiment", "—"),
                    "qa": row["qa"] if row.get("qa") is not None else "—",
                    "escalation": row.get("escalation", 0),
                    "trend": row.get("trend", "—"),
                }
                for row in overview_rows
            ]
            calls_table = ui.table(
                columns=[
                    {"name": "call_id", "label": "ID", "field": "call_id"},
                    {"name": "title", "label": "Titel", "field": "title"},
                    {"name": "sentiment", "label": "Sentiment", "field": "sentiment"},
                    {"name": "qa", "label": "QA", "field": "qa"},
                    {"name": "escalation", "label": "Eskalering", "field": "escalation"},
                    {"name": "trend", "label": "Trend", "field": "trend"},
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