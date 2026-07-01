"""Reusable empty-state panel for dashboard tabs."""

from __future__ import annotations

from collections.abc import Callable

from nicegui import ui


def render_empty_state(
    *,
    icon: str,
    title: str,
    hint: str,
    action: Callable[[], None] | None = None,
    action_label: str = "Försök igen",
) -> None:
    """Centered empty state with optional action button."""
    with ui.card().classes("w-full empty-state q-pa-lg"):
        with ui.column().classes("w-full items-center gap-2"):
            ui.icon(icon, size="lg").classes("text-grey")
            ui.label(title).classes("text-subtitle1")
            ui.label(hint).classes("text-caption text-grey text-center")
            if action:
                ui.button(action_label, on_click=action).props("outline dense")
