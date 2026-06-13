"""Tests for dashboard data layer (post–Fas 5 Streamlit removal)."""

from __future__ import annotations

from app.dashboard_launcher import resolve_dashboard_ui
from app.services.data_services import (
    _generate_fallback_reports,
    filter_reports,
    get_demo_transcripts,
    get_overall_sentiment,
)


def _fast_demo_reports() -> list[dict]:
    """Synthetic reports for unit tests (avoids slow full pipeline)."""
    return _generate_fallback_reports(get_demo_transcripts())


class TestDashboardLauncher:
    def test_resolve_dashboard_ui_default(self, monkeypatch):
        monkeypatch.delenv("DASHBOARD_UI", raising=False)
        assert resolve_dashboard_ui() == "nicegui"

    def test_resolve_dashboard_ui_explicit(self, monkeypatch):
        monkeypatch.setenv("DASHBOARD_UI", "nicegui")
        assert resolve_dashboard_ui() == "nicegui"

    def test_streamlit_deprecated_exits(self, monkeypatch):
        import pytest

        from app.dashboard_launcher import main

        monkeypatch.setenv("DASHBOARD_UI", "streamlit")
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1


class TestDataServicesDashboard:
    def test_get_demo_transcripts_count(self):
        transcripts = get_demo_transcripts()
        assert len(transcripts) == 5
        assert "segments" in transcripts[0]

    def test_generate_fallback_reports_shape(self):
        reports = _fast_demo_reports()
        assert len(reports) == 5
        assert "call_id" in reports[0]

    def test_filter_reports_sentiment(self):
        reports = _fast_demo_reports()
        filtered = filter_reports(reports, {"sentiment_filter": "all"})
        assert len(filtered) == len(reports)

    def test_get_overall_sentiment_shape(self):
        report = _fast_demo_reports()[0]
        sent = get_overall_sentiment(report)
        assert "label" in sent
        assert "score" in sent