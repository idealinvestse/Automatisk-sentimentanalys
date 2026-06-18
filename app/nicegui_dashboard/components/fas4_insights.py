"""Fas 4 Insikter tab – composes QA, insights, search and alerts sections.

Fas 3 – docs/GROK_BUILD_PLAN_FAS1-3.md
"""

from __future__ import annotations

from collections.abc import Callable

from nicegui import ui

from app.nicegui_dashboard.components.alerts_panel import render_alerts_panel
from app.nicegui_dashboard.components.insights_hot_topics import render_insights_section
from app.nicegui_dashboard.components.qa_scorecard import render_qa_scorecard_section
from app.nicegui_dashboard.state import DashboardState


def render_fas4_insights_tab(
    state: DashboardState,
    *,
    on_call_select: Callable[[str], None] | None = None,
    on_alerts_change: Callable[[], None] | None = None,
) -> Callable[[], None]:
    """Assemble Fas 4 insights tab. Returns combined refresh callback."""
    refreshers: list[Callable[[], None]] = []

    with ui.column().classes("w-full"):
        ui.label("🎯 Fas 4 Insikter").classes("text-h6 q-mb-md")
        ui.label(
            "QA-scorecard, hot topics, semantisk sökning och alerts – "
            "klicka för att öppna samtal i detaljvyn."
        ).classes("text-caption q-mb-lg")

        with ui.card().classes("w-full q-mb-md"):
            refreshers.append(
                render_qa_scorecard_section(state, on_call_select=on_call_select)
            )

        with ui.card().classes("w-full q-mb-md"):
            refreshers.append(render_insights_section(state, on_call_select=on_call_select))

        with ui.card().classes("w-full"):
            refreshers.append(
                render_alerts_panel(
                    state,
                    on_call_select=on_call_select,
                    on_dismiss_change=on_alerts_change,
                )
            )

    def refresh_all() -> None:
        for fn in refreshers:
            fn()

    return refresh_all