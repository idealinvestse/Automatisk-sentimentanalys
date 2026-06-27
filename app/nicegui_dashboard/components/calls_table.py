"""Filtered calls table with search, pagination, and row-click navigation.

Fas 6.1 – docs/MIGRATION_TO_NICEGUI_PLAN.md (paginering + sökning)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nicegui import ui

from app.nicegui_dashboard.components.empty_state import render_empty_state
from app.nicegui_dashboard.services.calls_filter import (
    format_search_hit_label,
    paginate_items,
    search_table_reports,
)
from app.nicegui_dashboard.services.demo_provider import reports_to_table_rows
from app.nicegui_dashboard.state import DashboardState

_QA_CELL_SLOT = """
<q-td :props="props">
  <span :class="props.row.qa_class">{{ props.value }}</span>
</q-td>
"""

_QA_STATUS_CELL_SLOT = """
<q-td :props="props">
  <q-chip
    v-if="props.value === 'Godkänd'"
    dense
    color="positive"
    text-color="white"
    :label="props.value"
  />
  <q-chip
    v-else-if="props.value === 'Underkänd'"
    dense
    color="negative"
    text-color="white"
    :label="props.value"
  />
  <span v-else class="text-grey">{{ props.value }}</span>
</q-td>
"""


def render_calls_table(
    state: DashboardState,
    *,
    reports: list[dict[str, Any]] | None = None,
    on_select: Callable[[str], None] | None = None,
) -> Callable[[], None]:
    """Render searchable, paginated calls table. Returns refresh callback."""

    @ui.refreshable
    def table_section() -> None:
        source = reports if reports is not None else state.reports
        searched = search_table_reports(source, state.table_search)
        page_items, total_pages, total_count = paginate_items(
            searched,
            state.table_page,
            state.table_page_size,
        )
        rows = reports_to_table_rows(page_items)
        query = (state.table_search or "").strip()

        ui.label("📋 Senaste samtal (klicka för detalj)").classes("text-subtitle2 q-mt-md")

        with ui.row().classes("w-full items-center gap-2 q-mb-sm flex-wrap"):
            ui.input(
                "Sök samtal",
                value=state.table_search,
                placeholder="call_id, ämne, agent eller transkripttext…",
                on_change=lambda e: _on_search(e.value or ""),
            ).classes("flex-grow").props("clearable dense")

            ui.select(
                label="Per sida",
                options=[10, 20, 50],
                value=state.table_page_size,
                on_change=lambda e: _on_page_size(int(e.value)),
            ).classes("w-24").props("dense")

        if query:
            ui.label(format_search_hit_label(total_count, query)).classes(
                "text-subtitle2 q-mb-xs"
            )

        if total_count == 0:
            if query:
                render_empty_state(
                    icon="search_off",
                    title="Inga träffar",
                    hint=f"Inget samtal matchar «{query}». Prova ett kortare sökord eller annat ämne.",
                )
            else:
                render_empty_state(
                    icon="inbox",
                    title="Inga samtal att visa",
                    hint="Ladda data från API eller justera filter.",
                )
        else:
            ui.label(
                f"Visar {len(rows)} av {total_count} samtal · sida {state.table_page}/{total_pages}"
            ).classes("text-caption q-mb-xs")

            table = ui.table(
                columns=[
                    {"name": "call_id", "label": "ID", "field": "call_id", "align": "left"},
                    {"name": "title", "label": "Ämne", "field": "title", "align": "left"},
                    {"name": "agent", "label": "Agent", "field": "agent", "align": "left"},
                    {"name": "category", "label": "Kategori", "field": "category", "align": "left"},
                    {"name": "sentiment", "label": "Sentiment", "field": "sentiment", "align": "left"},
                    {"name": "risk_level", "label": "Risk", "field": "risk_level", "align": "left"},
                    {"name": "alert_count", "label": "Aviseringar", "field": "alert_count", "align": "left"},
                    {"name": "qa_status", "label": "QA-status", "field": "qa_status", "align": "left"},
                    {"name": "qa_score", "label": "QA", "field": "qa_score", "align": "left"},
                ],
                rows=rows,
                row_key="call_id",
                pagination={"rowsPerPage": 0},
            ).classes("w-full")
            table.add_slot("body-cell-qa_score", _QA_CELL_SLOT)
            table.add_slot("body-cell-qa_status", _QA_STATUS_CELL_SLOT)

            def handle_row_click(e: Any) -> None:
                row = e.args[1] if len(e.args) > 1 else e.args[0]
                if isinstance(row, dict):
                    call_id = row.get("call_id")
                else:
                    call_id = str(row)
                if not call_id:
                    return
                state.selected_call_id = call_id
                ui.notify(f"Öppnar detalj för {call_id}")
                if on_select:
                    on_select(call_id)

            table.on("rowClick", handle_row_click)

            with ui.row().classes("w-full items-center justify-between q-mt-sm"):
                prev_btn = ui.button(
                    "◀ Föregående",
                    on_click=lambda: _change_page(state.table_page - 1, total_pages),
                ).props("flat dense")
                if state.table_page <= 1:
                    prev_btn.props("disable")
                ui.label(f"Sida {state.table_page} / {total_pages}").classes("text-caption")
                next_btn = ui.button(
                    "Nästa ▶",
                    on_click=lambda: _change_page(state.table_page + 1, total_pages),
                ).props("flat dense")
                if state.table_page >= total_pages:
                    next_btn.props("disable")

    def _on_search(value: str) -> None:
        state.table_search = value
        state.table_page = 1
        table_section.refresh()

    def _on_page_size(size: int) -> None:
        state.table_page_size = size
        state.table_page = 1
        table_section.refresh()

    def _change_page(new_page: int, total_pages: int) -> None:
        state.table_page = max(1, min(new_page, total_pages))
        table_section.refresh()

    table_section()
    return table_section.refresh