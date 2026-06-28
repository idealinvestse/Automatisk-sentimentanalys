"""Unified dashboard launcher with feature flag (Fas 5).

Fas 5 – docs/archive/MIGRATION_TO_NICEGUI_PLAN.md §3

Environment:
    DASHBOARD_UI=nicegui   (default, production)

Usage:
    python -m app.dashboard_launcher
"""

from __future__ import annotations

import os
import runpy
import sys


def resolve_dashboard_ui() -> str:
    """Return dashboard backend id from DASHBOARD_UI env (default: nicegui)."""
    return os.environ.get("DASHBOARD_UI", "nicegui").strip().lower()


def main() -> None:
    """Launch the configured dashboard UI."""
    ui = resolve_dashboard_ui()

    if ui not in {"nicegui", "ng"}:
        print(f"Okänt DASHBOARD_UI={ui!r}. Giltiga värden: nicegui", file=sys.stderr)
        raise SystemExit(2)

    runpy.run_module("app.nicegui_dashboard.main", run_name="__main__")


if __name__ == "__main__":
    main()