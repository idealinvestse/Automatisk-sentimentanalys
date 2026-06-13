"""NiceGUI Call Center Dashboard – entry point.

Fas 4 – docs/MIGRATION_TO_NICEGUI_PLAN.md §3
Polished UI, error handling, production-ready entry.

Run:
    pip install -e ".[dashboard-nicegui]"
    uvicorn src.api:app --port 8000
    python -m app.nicegui_dashboard.main

Docker:
    docker compose -f docker-compose.nicegui.yml up
"""

from __future__ import annotations

from nicegui import ui

from app.nicegui_dashboard.components.analytics_trends import render_analytics_tab
from app.nicegui_dashboard.components.call_detail import render_call_detail_tab
from app.nicegui_dashboard.components.layout import apply_dark_theme, render_header
from app.nicegui_dashboard.components.live_analysis import render_live_analysis_tab
from app.nicegui_dashboard.components.overview import render_overview_tab
from app.nicegui_dashboard.components.transcription_monitor import render_transcription_tab
from app.nicegui_dashboard.services.demo_provider import load_demo_reports, load_reports_from_api
from app.nicegui_dashboard.services.nicegui_api_client import APIError, NiceGUIAPIClient
from app.nicegui_dashboard.services.transcription_service import create_transcription_state
from app.nicegui_dashboard.services.ui_helpers import notify_error, notify_success, notify_warning
from app.nicegui_dashboard.state import DashboardState


@ui.page("/")
def dashboard_page() -> None:
    """Main dashboard page with tab navigation."""
    ui.page_title("Call Center Insights – NiceGUI Dashboard")
    dark = apply_dark_theme()

    api_client = NiceGUIAPIClient.from_env()
    reports = list(load_demo_reports())
    state = DashboardState(
        reports=reports,
        api_client=api_client,
        transcription=create_transcription_state(api_client=api_client),
    )

    refresh_header = render_header(state, phase_label="Produktion", dark_mode=dark)

    with ui.column().classes("w-full nicegui-dashboard"):
        _render_tabs(state, refresh_header)


def _render_tabs(state: DashboardState, refresh_header) -> None:
    with ui.tabs().classes("w-full") as tabs:
        overview_tab = ui.tab("Översikt")
        analytics_tab = ui.tab("Analys & Trender")
        detail_tab = ui.tab("Call Detail")
        trans_tab = ui.tab("Transkribering")
        live_tab = ui.tab("Live-analys")

    refresh_call_detail: list = []
    refresh_overview: list = []
    refresh_analytics: list = []

    def go_to_detail(call_id: str | None = None) -> None:
        if call_id:
            state.selected_call_id = call_id
        tabs.set_value(detail_tab)
        if refresh_call_detail:
            refresh_call_detail[0]()

    def show_example_detail() -> None:
        if state.reports:
            state.selected_call_id = state.reports[0].get("call_id")
        go_to_detail()

    async def reload_from_api() -> None:
        await load_from_api(notify=True)

    async def load_from_api(*, notify: bool = True) -> None:
        if not state.api_client:
            return
        try:
            if not await state.api_client.wait_for_health(attempts=3, interval=0.5):
                state.api_connected = False
                if refresh_header:
                    refresh_header()
                if notify:
                    notify_warning("Backend ej tillgänglig – använder demo-data")
                return
            state.api_connected = True
            state.reports = await load_reports_from_api(state.api_client)
            state.data_source = "api"
            if refresh_overview:
                refresh_overview[0]()
            if refresh_analytics:
                refresh_analytics[0]()
            if refresh_call_detail:
                refresh_call_detail[0]()
            if refresh_header:
                refresh_header()
            if notify:
                notify_success(f"Data laddad från API ({len(state.reports)} samtal)")
        except APIError as err:
            state.api_connected = False
            if refresh_header:
                refresh_header()
            if notify:
                notify_error(f"API-fel: {err}")
        except Exception as err:
            state.api_connected = False
            if refresh_header:
                refresh_header()
            if notify:
                notify_warning(f"Kunde inte ladda från API: {err}")

    with ui.tab_panels(tabs, value=overview_tab).classes("w-full"):
        with ui.tab_panel(overview_tab):
            refresh_fn = render_overview_tab(
                state,
                on_call_select=go_to_detail,
                on_show_example_detail=show_example_detail,
                on_reload_api=reload_from_api,
            )
            refresh_overview.append(refresh_fn)

        with ui.tab_panel(analytics_tab):
            refresh_fn = render_analytics_tab(state, on_call_select=go_to_detail)
            refresh_analytics.append(refresh_fn)

        with ui.tab_panel(detail_tab):
            refresh_fn = render_call_detail_tab(
                state,
                on_back=lambda: tabs.set_value(overview_tab),
            )
            refresh_call_detail.append(refresh_fn)

        with ui.tab_panel(trans_tab):
            if state.transcription:
                render_transcription_tab(state.transcription, api_client=state.api_client)

        with ui.tab_panel(live_tab):
            render_live_analysis_tab(state)

    async def initial_api_load() -> None:
        await load_from_api(notify=False)

    ui.timer(0.5, initial_api_load, once=True)


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        host="0.0.0.0",
        port=int(__import__("os").environ.get("NICEGUI_PORT", "8080")),
        title="Call Center NiceGUI Dashboard",
        reload=False,
        favicon="📞",
    )