"""Filtered calls table with row-click navigation.

Fas 2 – docs/MIGRATION_TO_NICEGUI_PLAN.md §3 (filtrerad calls-tabell med on_click)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nicegui import ui

from app.nicegui_dashboard.services.demo_provider import reports_to_table_rows
from app.nicegui_dashboard.state import DashboardState


def render_calls_table(
    state: DashboardState,
    *,
    reports: list[dict[str, Any]] | None = None,
    on_select: Callable[[str], None] | None = None,
) -> None:
    """Render the calls table; row click sets selected_call_id and triggers callback."""
    rows = reports_to_table_rows(reports if reports is not None else state.reports)

    ui.label("📋 Senaste samtal (klicka för detalj)").classes("text-subtitle2 q-mt-md")
    table = ui.table(
        columns=[
            {"name": "call_id", "label": "ID", "field": "call_id", "align": "left"},
            {"name": "title", "label": "Ämne", "field": "title", "align": "left"},
            {"name": "agent", "label": "Agent", "field": "agent", "align": "left"},
            {"name": "sentiment", "label": "Sentiment", "field": "sentiment", "align": "left"},
            {"name": "qa_score", "label": "QA", "field": "qa_score", "align": "left"},
        ],
        rows=rows,
        row_key="call_id",
    ).classes("w-full")

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