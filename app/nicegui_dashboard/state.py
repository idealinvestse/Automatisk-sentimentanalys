"""Session state for the NiceGUI dashboard.

Fas 3 – docs/archive/MIGRATION_TO_NICEGUI_PLAN.md §5 (State Management)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.nicegui_dashboard.services.nicegui_api_client import NiceGUIAPIClient
from app.nicegui_dashboard.services.transcription_service import TranscriptionState


@dataclass
class DashboardState:
    """Per-page dashboard state."""

    reports: list[dict[str, Any]]
    selected_call_id: str | None = None
    filters: dict[str, Any] = field(default_factory=dict)
    table_page: int = 1
    table_page_size: int = 20
    table_search: str = ""
    transcription: TranscriptionState | None = None
    api_client: NiceGUIAPIClient | None = None
    api_connected: bool = False
    data_source: str = "fallback"  # fallback | api | local_pipeline
    detail_source_tab: str = "overview"  # overview | analytics | fas4 | agent_performance
    selected_agent_id: str | None = None
    selected_qa_call_id: str | None = None
    semantic_search_query: str = ""
    dismissed_alert_keys: list[str] = field(default_factory=list)
