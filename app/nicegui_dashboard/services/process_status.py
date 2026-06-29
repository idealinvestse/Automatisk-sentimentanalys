"""Process status state for future ops dashboard (reads API or local JSONL)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from app.nicegui_dashboard.services.nicegui_api_client import NiceGUIAPIClient


def _default_events_path() -> Path:
    root = os.getenv("SENTIMENT_APP_ROOT", ".")
    return Path(root) / ".cache" / "process_events.jsonl"


class ProcessStatusState:
    """Stub service: load recent process events from local JSONL or API client."""

    def __init__(
        self,
        *,
        api_client: NiceGUIAPIClient | None = None,
        events_path: Path | None = None,
    ) -> None:
        self.api_client = api_client
        self.events_path = events_path or _default_events_path()

    def load_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Load recent status events from local JSONL file."""
        return self._load_from_file(limit=limit)

    async def load_recent_async(self, limit: int = 50) -> list[dict[str, Any]]:
        """Load recent status events from API with JSONL fallback."""
        if self.api_client is not None:
            try:
                data = await self.api_client.get_process_events(limit=limit)
                events = data.get("events")
                if isinstance(events, list):
                    return events
            except Exception:
                pass
        return self._load_from_file(limit=limit)

    def _load_from_file(self, *, limit: int) -> list[dict[str, Any]]:
        path = self.events_path
        if not path.is_file():
            return []
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = lines[-limit:] if limit > 0 else lines
        events: list[dict[str, Any]] = []
        for line in tail:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events
