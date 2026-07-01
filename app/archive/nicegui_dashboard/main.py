"""[DEPRECATED] NiceGUI Call Center Dashboard – entry point.

This dashboard has been superseded by the Next.js web UI in ``webui/``.
It is kept here for reference only and is no longer maintained.

See ``docs/WEBUI_MODERNIZATION_PLAN.md`` for the migration details.
To run the current dashboard, use ``cd webui && npm run dev``.

Original docstring follows:
Fas 4 – docs/archive/MIGRATION_TO_NICEGUI_PLAN.md §3
Polished UI, error handling, production-ready entry.

Run:
    pip install -e ".[dashboard-nicegui]"
    uvicorn src.api:app --port 8000
    python -m app.archive.nicegui_dashboard.main

Docker:
    docker compose -f docker-compose.nicegui.yml up
"""

from __future__ import annotations

from nicegui import ui

from app.archive.nicegui_dashboard.components.agent_performance import render_agent_performance_tab
from app.archive.nicegui_dashboard.components.analytics_trends import render_analytics_tab
from app.archive.nicegui_dashboard.components.call_detail import render_call_detail_tab
from app.archive.nicegui_dashboard.components.fas4_insights import render_fas4_insights_tab
from app.archive.nicegui_dashboard.components.layout import apply_dark_theme, render_header
from app.archive.nicegui_dashboard.components.onboarding import render_onboarding_banner
from app.archive.nicegui_dashboard.components.overview import render_overview_tab
from app.archive.nicegui_dashboard.components.test_lab import render_test_lab_tab
from app.archive.nicegui_dashboard.components.transcription_monitor import render_transcription_tab
from app.archive.nicegui_dashboard.services.demo_provider import (
    load_demo_reports,
    load_reports_from_api,
)
from app.archive.nicegui_dashboard.services.nicegui_api_client import APIError, NiceGUIAPIClient
from app.archive.nicegui_dashboard.services.report_ingest import append_report_to_state
from app.archive.nicegui_dashboard.services.transcription_service import create_transcription_state
from app.archive.nicegui_dashboard.services.ui_helpers import (
    notify_error,
    notify_success,
    notify_warning,
)
from app.archive.nicegui_dashboard.settings import is_dev_mode
from app.archive.nicegui_dashboard.state import DashboardState


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

    reload_ref: dict = {}

    async def header_reload() -> None:
        reload_fn = reload_ref.get("reload")
        if reload_fn:
            await reload_fn()

    refresh_header = render_header(
        state,
        phase_label="Produktion",
        dark_mode=dark,
        on_reload=header_reload,
    )

    with ui.column().classes("w-full nicegui-dashboard"):
        render_onboarding_banner()
        _render_tabs(state, refresh_header, reload_ref)


def _render_tabs(state: DashboardState, refresh_header, reload_ref: dict) -> None:
    with ui.tabs().classes("w-full") as tabs:
        overview_tab = ui.tab("Översikt")
        analytics_tab = ui.tab("Analys & Trender")
        agent_tab = ui.tab("Agentprestanda")
        fas4_tab = ui.tab("Fas 4 Insikter")
        detail_tab = ui.tab("Samtalsdetalj")
        trans_tab = ui.tab("Transkribering")
        test_tab = ui.tab("Testlabb") if is_dev_mode() else None

    refresh_call_detail: list = []
    refresh_overview: list = []
    refresh_analytics: list = []
    refresh_agent: list = []
    refresh_fas4: list = []

    def go_to_detail(
        call_id: str | None = None,
        *,
        source: str = "overview",
    ) -> None:
        if call_id:
            state.selected_call_id = call_id
        state.detail_source_tab = source
        tabs.set_value(detail_tab)
        if refresh_call_detail:
            refresh_call_detail[0]()

    def show_example_detail() -> None:
        if state.reports:
            state.selected_call_id = state.reports[0].get("call_id")
        go_to_detail(source="overview")

    def go_back_from_detail() -> None:
        source = state.detail_source_tab
        if source == "analytics":
            target = analytics_tab
        elif source == "agent_performance":
            target = agent_tab
        elif source == "fas4":
            target = fas4_tab
        else:
            target = overview_tab
        tabs.set_value(target)

    def go_to_agent(agent_id: str) -> None:
        state.selected_agent_id = agent_id
        tabs.set_value(agent_tab)
        if refresh_agent:
            refresh_agent[0]()

    def go_to_overview() -> None:
        tabs.set_value(overview_tab)
        if refresh_overview:
            refresh_overview[0]()

    def on_transcription_report(report: dict) -> None:
        cid = append_report_to_state(state, report)
        notify_success(f"Rapport {cid} tillagd i dashboard")
        go_to_detail(cid, source="overview")

    async def reload_from_api() -> None:
        await load_from_api(notify=True)

    reload_ref["reload"] = reload_from_api

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
            if refresh_agent:
                refresh_agent[0]()
            if refresh_fas4:
                refresh_fas4[0]()
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
                on_call_select=lambda cid: go_to_detail(cid, source="overview"),
                on_agent_select=go_to_agent,
                on_show_example_detail=show_example_detail,
                on_reload_api=reload_from_api if state.api_client else None,
            )
            refresh_overview.append(refresh_fn)

        with ui.tab_panel(analytics_tab):
            refresh_fn = render_analytics_tab(
                state,
                on_call_select=lambda cid: go_to_detail(cid, source="analytics"),
            )
            refresh_analytics.append(refresh_fn)

        with ui.tab_panel(agent_tab):
            refresh_fn = render_agent_performance_tab(
                state,
                on_call_select=lambda cid: go_to_detail(cid, source="agent_performance"),
            )
            refresh_agent.append(refresh_fn)

        with ui.tab_panel(fas4_tab):
            refresh_fn = render_fas4_insights_tab(
                state,
                on_call_select=lambda cid: go_to_detail(cid, source="fas4"),
                on_alerts_change=refresh_header,
                on_topic_filter=go_to_overview,
            )
            refresh_fas4.append(refresh_fn)

        with ui.tab_panel(detail_tab):
            refresh_fn = render_call_detail_tab(state, on_back=go_back_from_detail)
            refresh_call_detail.append(refresh_fn)

        with ui.tab_panel(trans_tab):
            if state.transcription:
                render_transcription_tab(
                    state.transcription,
                    api_client=state.api_client,
                    on_report_ready=on_transcription_report,
                )

        if test_tab is not None:
            with ui.tab_panel(test_tab):
                render_test_lab_tab(state)

    async def initial_api_load() -> None:
        await load_from_api(notify=False)

    ui.timer(0.5, initial_api_load, once=True)


if __name__ in {"__main__", "__mp_main__"}:
    from src.core.logging_config import configure_logging

    configure_logging()
    storage_secret = __import__("os").environ.get("NICEGUI_STORAGE_SECRET")
    if not storage_secret:
        raise ValueError(
            "NICEGUI_STORAGE_SECRET environment variable must be set for production security. "
            "Generate one with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
        )
    ui.run(
        host="0.0.0.0",
        port=int(__import__("os").environ.get("NICEGUI_PORT", "8080")),
        title="Call Center NiceGUI Dashboard",
        reload=False,
        favicon="📞",
        storage_secret=storage_secret,
    )
