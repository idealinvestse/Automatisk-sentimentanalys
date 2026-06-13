"""Dashboard theme: custom CSS and dark mode.

Fas 4 – docs/MIGRATION_TO_NICEGUI_PLAN.md §3
"""

from __future__ import annotations

from nicegui import ui

# Fas 4 – modern dark dashboard styling (Tailwind-compatible via Quasar)
DASHBOARD_CSS = """
.nicegui-dashboard .q-card {
    border-radius: 10px;
}
.nicegui-dashboard .q-table {
    border-radius: 8px;
}
.nicegui-dashboard .log-info { color: #4ade80; }
.nicegui-dashboard .log-warning { color: #fbbf24; }
.nicegui-dashboard .log-error { color: #f87171; }
.nicegui-dashboard .api-status-connected { color: #4ade80; }
.nicegui-dashboard .api-status-offline { color: #fbbf24; }
"""


def apply_dashboard_theme() -> ui.dark_mode:
    """Enable dark mode, inject CSS, return dark_mode handle for toggle."""
    dark = ui.dark_mode()
    dark.enable()
    ui.add_css(DASHBOARD_CSS)
    return dark