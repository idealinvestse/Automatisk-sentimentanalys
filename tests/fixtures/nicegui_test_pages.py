"""Isolated NiceGUI pages for dashboard component rendering tests.

Fas 6.1 – docs/archive/MIGRATION_TO_NICEGUI_PLAN.md (utökade tester)
Executed via pytest `user` fixture + @pytest.mark.module_under_test
"""

from __future__ import annotations

from nicegui import ui

from app.nicegui_dashboard.components.agent_performance import render_agent_performance_tab
from app.nicegui_dashboard.components.analytics_trends import render_analytics_tab
from app.nicegui_dashboard.components.call_detail import render_call_detail_tab
from app.nicegui_dashboard.components.fas4_insights import render_fas4_insights_tab
from app.nicegui_dashboard.components.llm_judge_panel import (
    render_llm_judge_panel,
    render_llm_judge_summary,
)
from app.nicegui_dashboard.components.onboarding import render_onboarding_banner
from app.nicegui_dashboard.components.overview import render_overview_tab
from app.nicegui_dashboard.components.test_lab import render_test_lab_tab
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


@ui.page("/overview-search")
def _overview_search_page() -> None:
    state = DashboardState(reports=_REPORTS, table_search="__no_match_xyz__")
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


@ui.page("/agent-performance")
def _agent_performance_page() -> None:
    state = DashboardState(reports=_REPORTS)
    render_agent_performance_tab(state, on_call_select=lambda _cid: None)


@ui.page("/fas4-insights")
def _fas4_insights_page() -> None:
    state = DashboardState(reports=_REPORTS)
    render_fas4_insights_tab(state, on_call_select=lambda _cid: None)


@ui.page("/test-lab")
def _test_lab_page() -> None:
    state = DashboardState(reports=_REPORTS, api_client=None)
    render_test_lab_tab(state)


@ui.page("/onboarding")
def _onboarding_page() -> None:
    render_onboarding_banner()


@ui.page("/llm-judge-empty")
def _llm_judge_empty_page() -> None:
    render_llm_judge_panel(None)


@ui.page("/llm-judge-data")
def _llm_judge_data_page() -> None:
    render_llm_judge_panel(
        {
            "verdicts": [
                {
                    "segment_index": 2,
                    "original_sentiment": "neutral",
                    "original_confidence": 0.45,
                    "judge_label": "negative",
                    "judge_confidence": 0.82,
                    "reasoning": "Kund nämner frustration.",
                }
            ]
        }
    )


@ui.page("/llm-judge-summary")
def _llm_judge_summary_page() -> None:
    render_llm_judge_summary(
        {
            "verdicts": [
                {"original_sentiment": "neutral", "judge_label": "negative"},
                {"original_sentiment": "negative", "judge_label": "negative"},
            ]
        }
    )


@ui.page("/")
def _index() -> None:
    ui.label("Dashboard test harness")


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(storage_secret="nicegui-test-secret", show=False)
