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
    """Load recent process events from local JSONL or API client."""

    def __init__(
        self,
        *,
        api_client: NiceGUIAPIClient | None = None,
        events_path: Path | None = None,
    ) -> None:
        self.api_client = api_client
        self.events_path = events_path or _default_events_path()

    def load_recent(self, limit: int = 50, **filters: Any) -> list[dict[str, Any]]:
        """Load recent status events from local JSONL file."""
        return self._filter_events(self._load_from_file(limit=max(limit, 500)), limit, **filters)

    async def load_recent_async(self, limit: int = 50, **filters: Any) -> list[dict[str, Any]]:
        """Load recent status events from API with JSONL fallback."""
        if self.api_client is not None:
            try:
                data = await self.api_client.get_process_events(limit=limit, **filters)
                events = data.get("events")
                if isinstance(events, list):
                    return events
            except Exception:
                pass
        return self.load_recent(limit=limit, **filters)

    async def job_status_async(self, job_id: str) -> dict[str, Any]:
        """Fetch live job summary from API."""
        if self.api_client is not None:
            try:
                return await self.api_client.get_job_status(job_id)
            except Exception:
                pass
        from src.core.status import derive_job_status

        events = self.load_recent(limit=1000, job_id=job_id)
        return derive_job_status(events, job_id)

    @staticmethod
    def _filter_events(
        events: list[dict[str, Any]],
        limit: int,
        *,
        job_id: str | None = None,
        component: str | None = None,
        level: str | None = None,
        since: str | None = None,
    ) -> list[dict[str, Any]]:
        if job_id:
            events = [e for e in events if e.get("job_id") == job_id]
        if component:
            events = [e for e in events if e.get("component") == component]
        if level:
            events = [e for e in events if e.get("level") == level.upper()]
        if since:
            events = [e for e in events if e.get("ts", "") >= since]
        if limit > 0:
            events = events[-limit:]
        return events

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
