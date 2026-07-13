"""Compatibility shim — prefer `python -m launcher.dashboard_spawn`."""

from __future__ import annotations

from launcher.dashboard_spawn import main, resolve_dashboard_ui

__all__ = ["main", "resolve_dashboard_ui"]

if __name__ == "__main__":
    main()
