"""NiceGUI User-simulation tests for dashboard component rendering.

Uses the NiceGUI pytest `user` fixture (no browser required).
"""

from __future__ import annotations

import pytest
from nicegui import ui

_NICEGUI_MAIN = "tests/fixtures/nicegui_test_pages.py"

pytestmark = pytest.mark.nicegui_main_file(_NICEGUI_MAIN)


async def test_overview_renders_kpi_and_table(user) -> None:
    await user.open("/overview")
    await user.should_see("Översikt")
    await user.should_see("Senaste samtal")
    await user.should_see("Visar 5 av 5 samtal")
    await user.should_see(ui.table)


async def test_overview_search_and_pagination_controls(user) -> None:
    await user.open("/overview")
    await user.should_see("Sök samtal")
    await user.should_see("Föregående")
    await user.should_see("Nästa")


async def test_overview_search_shows_no_hits_empty_state(user) -> None:
    await user.open("/overview-search")
    await user.should_see("Inga träffar")
    await user.should_see("__no_match_xyz__")


async def test_call_detail_renders_transcript(user) -> None:
    await user.open("/call-detail")
    await user.should_see("Samtalsdetalj")
    await user.should_see("Transkript")
    await user.should_see("Strukturerade insikter")


async def test_call_detail_large_virtualized(user) -> None:
    await user.open("/call-detail-large")
    await user.should_see("virtualiserad")
    await user.should_see("Virtualiserat transkript")


async def test_transcription_monitor_renders(user) -> None:
    await user.open("/transcription")
    await user.should_see("Transkriberingskö")
    await user.should_see("Starta")
    await user.should_see("Återanslut WS")
    await user.should_see("Händelselogg")
    await user.should_see(ui.table)


async def test_analytics_tab_renders_charts(user) -> None:
    await user.open("/analytics")
    await user.should_see("Analys & trender")
    await user.should_see("Sentimentfördelning")
    await user.should_see("Kundsentiment över tid")
    await user.should_see("Agentprestanda över tid")
    await user.should_see("Heta ämnen")
    await user.should_see("Detaljerad samtalsöversikt")
    await user.should_see("Exportera CSV")
    await user.should_see("Filtrera agent")
    await user.should_see(ui.table)


async def test_test_lab_renders_sections(user) -> None:
    await user.open("/test-lab")
    await user.should_see("Testlabb")
    await user.should_see("Ljudprover")
    await user.should_see("Text & pipeline")
    await user.should_see("System")
    await user.should_see("Rapporter")
    await user.should_see("Kör hälsokontroll")
    await user.should_see("Uppdatera lista")
    await user.should_see(ui.table)


async def test_onboarding_banner_renders(user) -> None:
    await user.open("/onboarding")
    await user.should_see("Kom igång")
    await user.should_see("Samtalsdetalj")


async def test_llm_judge_panel_empty_state(user) -> None:
    await user.open("/llm-judge-empty")
    await user.should_see("LLM Judge")
    await user.should_see("Inga LLM-judge bedömningar")


async def test_llm_judge_panel_with_data(user) -> None:
    await user.open("/llm-judge-data")
    await user.should_see("segment bedömdes")
    await user.should_see("Motivering")


async def test_llm_judge_summary_badge(user) -> None:
    await user.open("/llm-judge-summary")
    await user.should_see("LLM Judge:")