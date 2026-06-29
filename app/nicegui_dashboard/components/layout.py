"""Layout helpers: header, theme, API status.

Fas 4 – docs/archive/MIGRATION_TO_NICEGUI_PLAN.md §3
"""

from __future__ import annotations

from collections.abc import Callable

from nicegui import ui

from app.nicegui_dashboard.components.alerts_panel import count_active_alerts
from app.nicegui_dashboard.components.theme import apply_dashboard_theme
from app.nicegui_dashboard.state import DashboardState


def apply_dark_theme() -> ui.dark_mode:
    """Enable dark mode + custom CSS. Returns dark_mode for toggle."""
    return apply_dashboard_theme()


def render_header(
    state: DashboardState | None = None,
    *,
    phase_label: str = "Fas 4",
    dark_mode: ui.dark_mode | None = None,
    on_reload: Callable[[], None] | None = None,
) -> Callable[[], None] | None:
    """Render header; returns callback to refresh API status label."""
    status_label: ui.label | None = None
    alerts_badge: ui.badge | None = None

    with ui.header(elevated=True).classes("items-center justify-between nicegui-dashboard"):
        with ui.row().classes("items-center gap-2"):
            ui.label("📞 Svensk Call Center – Samtalsintelligens").classes("text-h5")
            ui.label(f"| {phase_label}").classes("text-caption text-grey")

        with ui.row().classes("items-center gap-3"):
            if state:
                n_alerts = count_active_alerts(state)
                with ui.row().classes("items-center gap-1"):
                    ui.icon("notifications_active", size="sm")
                    alerts_badge = ui.badge(
                        str(n_alerts) if n_alerts > 0 else "0",
                        color="negative" if n_alerts > 0 else "grey",
                    )

            if state and state.api_client:
                cls = "api-status-connected" if state.api_connected else "api-status-offline"
                txt = "API ●" if state.api_connected else "API ○"
                status_label = ui.label(f"{txt} {state.api_client.base_url}").classes(
                    f"text-caption {cls}"
                )
                if on_reload:
                    ui.button(icon="refresh", on_click=on_reload).props("flat round dense").tooltip(
                        "Ladda om data från API"
                    )

            if dark_mode is not None:
                ui.button(
                    icon="dark_mode",
                    on_click=dark_mode.toggle,
                ).props(
                    "flat round dense"
                ).tooltip("Växla ljust/mörkt tema")

    def refresh_status() -> None:
        if status_label and state and state.api_client:
            txt = "API ●" if state.api_connected else "API ○"
            status_label.set_text(f"{txt} {state.api_client.base_url}")
        if alerts_badge and state:
            n_alerts = count_active_alerts(state)
            alerts_badge.set_text(str(n_alerts))
            alerts_badge.props(f"color={'negative' if n_alerts > 0 else 'grey'}")

    return refresh_status if (status_label or alerts_badge) else None
