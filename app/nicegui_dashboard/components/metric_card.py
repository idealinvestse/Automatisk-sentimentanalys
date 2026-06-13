"""Reusable metric card (ui.card + labels) for dashboards without ui.metric."""

from __future__ import annotations

from nicegui import ui
from nicegui.elements.label import Label


def metric_card(label: str, value: str | int | float, *, color: str = "primary") -> Label:
    """Render a metric-style card; returns the value label for live updates."""
    with ui.card().classes("flex-1"):
        ui.label(label).classes("text-caption")
        return ui.label(str(value)).classes(f"text-h4 text-{color}")