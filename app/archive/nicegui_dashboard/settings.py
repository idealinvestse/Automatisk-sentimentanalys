"""Dashboard runtime settings (env-driven)."""

from __future__ import annotations

import os


def is_dev_mode() -> bool:
    """True when SENTIMENT_DEV_MODE is set (shows Testlabb tab etc.)."""
    return os.environ.get("SENTIMENT_DEV_MODE", "").lower() in ("1", "true", "yes")


def ws_status_label(status: str) -> str:
    """Swedish WebSocket connection status label."""
    return {
        "connected": "Ansluten",
        "reconnecting": "Återansluter",
        "disconnected": "Frånkopplad",
    }.get(status, "Frånkopplad")
