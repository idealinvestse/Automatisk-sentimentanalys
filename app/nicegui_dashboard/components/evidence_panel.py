"""Evidence quotes panel for call detail and QA views."""

from __future__ import annotations

from typing import Any

from nicegui import ui

from app.nicegui_dashboard.components.empty_state import render_empty_state
from app.nicegui_dashboard.components.ui_primitives import render_section_title
from app.services.data_services import get_evidence_quotes


def render_evidence_panel(report: dict[str, Any] | None, *, max_quotes: int = 8) -> None:
    """Render deduplicated evidence quotes from pipeline report."""
    if not report:
        return
    quotes = get_evidence_quotes(report, max_quotes=max_quotes)
    with ui.expansion("Beviscitat", icon="format_quote", value=False).classes("w-full"):
        if not quotes:
            render_empty_state(
                icon="format_quote",
                title="Inga beviscitat",
                hint="Kör pipeline med LLM/QA för att samla citat och spans.",
            )
            return
        for i, quote in enumerate(quotes, start=1):
            with ui.card().classes("q-mb-xs q-pa-sm"):
                ui.label(f"{i}. {quote}").classes("text-body2")