"""Data services for NiceGUI dashboard."""

from app.archive.nicegui_dashboard.services.demo_provider import (
    load_demo_reports,
    load_reports_from_api,
    reports_to_table_rows,
)
from app.archive.nicegui_dashboard.services.nicegui_api_client import APIError, NiceGUIAPIClient

__all__ = [
    "APIError",
    "NiceGUIAPIClient",
    "load_demo_reports",
    "load_reports_from_api",
    "reports_to_table_rows",
]
