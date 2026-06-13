"""NiceGUI User-simulation tests for dashboard component rendering.

Fas 6.1 – docs/MIGRATION_TO_NICEGUI_PLAN.md (utökade tester)
Uses user_simulation context (no browser required).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from nicegui import ui

from nicegui.testing.user_simulation import user_simulation

_MAIN = (Path(__file__).parent / "fixtures" / "nicegui_test_pages.py").resolve()


@pytest.mark.asyncio
async def test_overview_renders_kpi_and_table() -> None:
    async with user_simulation(main_file=_MAIN) as user:
        await user.open("/overview")
        await user.should_see("Översikt")
        await user.should_see("Senaste samtal")
        await user.should_see("Visar 5 av 5 samtal")
        await user.should_see(ui.table)


@pytest.mark.asyncio
async def test_overview_search_and_pagination_controls() -> None:
    async with user_simulation(main_file=_MAIN) as user:
        await user.open("/overview")
        await user.should_see("Sök samtal")
        await user.should_see("Föregående")
        await user.should_see("Nästa")


@pytest.mark.asyncio
async def test_call_detail_renders_transcript() -> None:
    async with user_simulation(main_file=_MAIN) as user:
        await user.open("/call-detail")
        await user.should_see("Call Detail")
        await user.should_see("Transkript")
        await user.should_see("Structured Insights")


@pytest.mark.asyncio
async def test_call_detail_large_virtualized() -> None:
    async with user_simulation(main_file=_MAIN) as user:
        await user.open("/call-detail-large")
        await user.should_see("virtualiserad")
        await user.should_see("Virtualiserat transkript")


@pytest.mark.asyncio
async def test_transcription_monitor_renders() -> None:
    async with user_simulation(main_file=_MAIN) as user:
        await user.open("/transcription")
        await user.should_see("Transkriberingskö")
        await user.should_see("Start batch")
        await user.should_see("Reconnect WS")
        await user.should_see("Test loggrad")


@pytest.mark.asyncio
async def test_analytics_tab_renders_charts() -> None:
    async with user_simulation(main_file=_MAIN) as user:
        await user.open("/analytics")
        await user.should_see("Analys & Trender")
        await user.should_see("Customer sentiment trajectory")
        await user.should_see("Agent performance trends")
        await user.should_see("Hot topics")