"""Isolated NiceGUI pages for dashboard component rendering tests.

Fas 6.1 – docs/MIGRATION_TO_NICEGUI_PLAN.md (utökade tester)
Executed via pytest User fixture + @pytest.mark.nicegui_main_file
"""

from __future__ import annotations

from nicegui import ui

from app.nicegui_dashboard.components.analytics_trends import render_analytics_tab
from app.nicegui_dashboard.components.call_detail import render_call_detail_tab
from app.nicegui_dashboard.components.overview import render_overview_tab
from app.nicegui_dashboard.components.transcription_monitor import render_transcription_tab
from app.nicegui_dashboard.services.demo_provider import load_demo_reports
from app.nicegui_dashboard.services.transcript_virtualizer import make_synthetic_segments
from app.nicegui_dashboard.services.transcription_service import TranscriptionState
from app.nicegui_dashboard.state import DashboardState

_REPORTS = list(load_demo_reports())


@ui.page("/overview")
def _overview_page() -> None:
    state = DashboardState(reports=_REPORTS)
    render_overview_tab(state, on_call_select=lambda _cid: None)


@ui.page("/call-detail")
def _call_detail_page() -> None:
    state = DashboardState(
        reports=_REPORTS,
        selected_call_id=_REPORTS[0].get("call_id"),
    )
    render_call_detail_tab(state)


@ui.page("/call-detail-large")
def _call_detail_large_page() -> None:
    large = {
        "call_id": "CALL-LARGE",
        "title": "Stort test-samtal",
        "meta": {"agent": "Test-Agent", "duration_s": 3600},
        "segments": make_synthetic_segments(120),
        "sentiment_results": [{"label": "neutral", "score": 0.5} for _ in range(120)],
        "results": {"qa": {"overall_qa_score": 80}},
    }
    state = DashboardState(reports=[large], selected_call_id="CALL-LARGE")
    render_call_detail_tab(state)


@ui.page("/transcription")
def _transcription_page() -> None:
    trans = TranscriptionState()
    trans.logs = [
        {
            "ts": "12:00:00",
            "level": "INFO",
            "msg": "Test loggrad",
            "source": "local",
        }
    ]
    render_transcription_tab(trans, api_client=None)


@ui.page("/analytics")
def _analytics_page() -> None:
    state = DashboardState(reports=_REPORTS)
    render_analytics_tab(state, on_call_select=lambda _cid: None)


@ui.page("/")
def _index() -> None:
    ui.label("Dashboard test harness")


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(storage_secret="nicegui-test-secret", show=False)