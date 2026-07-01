"""Insights, hot topics and semantic search section.

Fas 3 – docs/archive/GROK_BUILD_PLAN_FAS1-3.md
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nicegui import ui

from app.archive.nicegui_dashboard.components.empty_state import render_empty_state
from app.archive.nicegui_dashboard.components.hot_topic_wordcloud import render_hot_topics_wordcloud
from app.archive.nicegui_dashboard.services.fas4_data import (
    fetch_hot_topics,
    fetch_semantic_search,
    format_evidence_spans,
    list_agent_ids,
    local_hot_topics_detailed,
    resolve_call_id_from_hit,
)
from app.archive.nicegui_dashboard.services.ui_helpers import notify_api_error
from app.archive.nicegui_dashboard.state import DashboardState
from app.services.data_services import filter_reports


def render_insights_section(
    state: DashboardState,
    *,
    on_call_select: Callable[[str], None] | None = None,
    on_topic_select: Callable[[str], None] | None = None,
) -> Callable[[], None]:
    """Hot topics table + semantic search with ranked results."""

    topics_source = {"value": "local"}
    search_source = {"value": "local"}
    topics_cache: dict[str, Any] = {"rows": [], "loaded": False}

    def _reports() -> list[dict[str, Any]]:
        return filter_reports(state.reports, state.filters)

    async def _load_topics() -> None:
        reports = _reports()
        if not reports:
            return
        try:
            if state.api_client and state.api_connected:
                topics, src = await fetch_hot_topics(state.api_client, reports)
                topics_source["value"] = src
            else:
                topics = local_hot_topics_detailed(reports)
                topics_source["value"] = "local"
        except Exception as err:
            notify_api_error(err)
            topics = local_hot_topics_detailed(reports)
            topics_source["value"] = "local"

        topics_cache["rows"] = [
            {
                "topic": t.get("topic", "—"),
                "volume": t.get("volume", 0),
                "sentiment": (
                    f"{t['avg_sentiment']:.2f}" if t.get("avg_sentiment") is not None else "—"
                ),
                "trend": t.get("trend", "—"),
                "evidence": format_evidence_spans(t.get("evidence_spans") or []),
            }
            for t in topics
        ]
        topics_cache["loaded"] = True
        insights_section.refresh()

    @ui.refreshable
    def insights_section() -> None:
        reports = _reports()
        ui.label("💡 Insikter & heta ämnen").classes("text-subtitle1 q-mb-sm")

        if not reports:
            ui.label("Ingen data för insikter.").classes("text-caption")
            return

        ui.label(f"Hot topics: {topics_source['value']}").classes("text-caption q-mb-xs")
        rows = topics_cache["rows"]
        if rows:
            topics_table = ui.table(
                columns=[
                    {"name": "topic", "label": "Ämne", "field": "topic"},
                    {"name": "volume", "label": "Volym", "field": "volume"},
                    {"name": "sentiment", "label": "Sentiment", "field": "sentiment"},
                    {"name": "trend", "label": "Trend", "field": "trend"},
                    {"name": "evidence", "label": "Bevis", "field": "evidence"},
                ],
                rows=rows,
                row_key="topic",
            ).classes("w-full")

            def _topic_row_click(e: Any) -> None:
                row = e.args[1] if len(e.args) > 1 else e.args[0]
                topic = row.get("topic") if isinstance(row, dict) else None
                if not topic:
                    return
                state.filters["topic_filter"] = str(topic).lower()
                if on_topic_select:
                    on_topic_select(str(topic))
                ui.notify(f"Filter på ämne: {topic}", type="info")
                insights_section.refresh()

            topics_table.on("rowClick", _topic_row_click)
            ui.label("Klicka på ett ämne för att filtrera samtal.").classes(
                "text-caption text-grey q-mt-xs"
            )
            with ui.card().classes("w-full q-mt-sm"):
                render_hot_topics_wordcloud(reports)
        elif topics_cache["loaded"]:
            ui.label("Inga hot topics hittades.").classes("text-caption")
        else:
            ui.label("Laddar hot topics...").classes("text-caption")

        ui.separator().classes("q-my-md")
        ui.label("🔍 Semantisk sökning").classes("text-subtitle1 q-mb-sm")

        agents = ["Alla"] + list_agent_ids(reports)
        agent_filter_val = {"current": None}

        with ui.row().classes("w-full gap-3 flex-wrap items-end"):
            search_input = (
                ui.input(
                    "Sökfråga",
                    value=state.semantic_search_query,
                    placeholder="t.ex. kunder klagade på faktura och låg empati",
                )
                .classes("flex-grow")
                .props("dense clearable")
            )
            ui.select(
                options=agents,
                value="Alla",
                label="Agent-filter",
                on_change=lambda e: agent_filter_val.update(
                    {"current": None if e.value == "Alla" else e.value}
                ),
            ).classes("min-w-36").props("dense")

        results_container = ui.column().classes("w-full q-mt-sm")
        search_lbl = ui.label("").classes("text-caption")

        async def _run_search() -> None:
            query = (search_input.value or "").strip()
            state.semantic_search_query = query
            if not query:
                ui.notify("Ange en sökfråga", type="warning")
                return

            search_lbl.set_text("Söker...")
            results_container.clear()
            try:
                if state.api_client and state.api_connected:
                    hits, src = await fetch_semantic_search(
                        state.api_client,
                        query,
                        reports,
                        top_k=8,
                        agent_filter=agent_filter_val["current"],
                    )
                    search_source["value"] = src
                else:
                    from app.archive.nicegui_dashboard.services.fas4_data import (
                        local_semantic_search,
                    )

                    hits = local_semantic_search(
                        query,
                        reports,
                        top_k=8,
                        agent_filter=agent_filter_val["current"],
                    )
                    search_source["value"] = "local"
            except Exception as err:
                notify_api_error(err)
                hits = []
                search_source["value"] = "local"

            with results_container:
                if not hits:
                    render_empty_state(
                        icon="search_off",
                        title="Inga träffar",
                        hint="Prova en annan sökfråga eller justera agentfiltret.",
                    )
                else:
                    for hit in hits:
                        if not isinstance(hit, dict):
                            continue
                        cid = resolve_call_id_from_hit(hit, reports) or hit.get("call_id", "?")
                        highlights = hit.get("highlights") or []
                        hl_text = " · ".join(str(h)[:100] for h in highlights[:2]) or "—"
                        with ui.card().classes("w-full q-mb-sm"):
                            with ui.row().classes("w-full items-center justify-between"):
                                ui.label(f"{cid} (score {hit.get('score', 0):.2f})").classes(
                                    "text-subtitle2"
                                )
                                if on_call_select:
                                    ui.button(
                                        "Öppna",
                                        on_click=lambda c=cid: on_call_select(c),
                                    ).props("flat dense")
                            ui.label(hl_text).classes("text-caption")
                            ev = format_evidence_spans(hit.get("evidence_spans") or [])
                            if ev != "—":
                                ui.label(ev).classes("text-caption text-grey")

            search_lbl.set_text(f"Sökning: {search_source['value']} · {len(hits)} träffar")

        ui.button("Sök", on_click=_run_search).props("color=primary")

    def refresh_all() -> None:
        insights_section.refresh()
        ui.timer(0.05, _load_topics, once=True)

    insights_section()
    ui.timer(0.1, _load_topics, once=True)
    return refresh_all
