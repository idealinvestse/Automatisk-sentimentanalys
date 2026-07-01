"""Reusable metric card (ui.card + labels) for dashboards without ui.metric."""

from __future__ import annotations

from nicegui.elements.label import Label

from app.archive.nicegui_dashboard.components.ui_primitives import metric_card as _metric_card


def metric_card(
    label: str,
    value: str | int | float,
    *,
    color: str = "primary",
    size: str = "default",
) -> Label:
    """Render a metric-style card; returns the value label for live updates."""
    return _metric_card(label, value, color=color, size=size)
