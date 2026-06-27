"""Shared UI primitives for consistent dashboard layout and typography."""

from __future__ import annotations

from nicegui import ui
from nicegui.elements.label import Label


def render_tab_header(
    title: str,
    *,
    hint: str | None = None,
    meta: str | None = None,
) -> None:
    """Tab-level header: title row with optional hint and meta caption."""
    with ui.row().classes("w-full items-center justify-between q-mb-md dashboard-tab"):
        ui.label(title).classes("text-h6")
        if meta:
            ui.label(meta).classes("text-caption text-grey")
    if hint:
        ui.label(hint).classes("text-caption text-grey q-mb-md")


def render_section_title(title: str, *, icon: str | None = None) -> None:
    """Section-level title within a tab."""
    classes = "text-subtitle1 q-mb-sm dashboard-section"
    if icon:
        with ui.row().classes("items-center gap-1 " + classes):
            ui.icon(icon, size="sm")
            ui.label(title)
    else:
        ui.label(title).classes(classes)


def metric_card(
    label: str,
    value: str | int | float,
    *,
    color: str = "primary",
    size: str = "default",
) -> Label:
    """Render a metric card; returns the value label for live updates."""
    value_class = "text-h4" if size == "default" else "text-h6"
    with ui.card().classes("flex-1 min-w-card"):
        ui.label(label).classes("text-caption text-grey")
        return ui.label(str(value)).classes(f"{value_class} text-{color}")