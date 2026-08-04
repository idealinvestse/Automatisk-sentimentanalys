"""File-backed store for analyzed call reports (server-side call history)."""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CallStore:
    """Persist call analysis artifacts as JSON under ``{state_dir}/calls/``."""

    def __init__(self, state_dir: str | Path) -> None:
        self._root = Path(state_dir) / "calls"
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, call_id: str) -> Path:
        if not _SAFE_ID.match(call_id):
            raise ValueError(f"Invalid call id: {call_id!r}")
        return self._root / f"{call_id}.json"

    def save(self, call_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Upsert a call record. Returns the stored document."""
        path = self._path(call_id)
        doc = {
            "id": call_id,
            "created_at": payload.get("created_at") or _utc_now(),
            "updated_at": _utc_now(),
            "transcript": payload.get("transcript") or {},
            "report": payload.get("report") or {},
            "meta": payload.get("meta") or {},
        }
        with self._lock:
            path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        return doc

    def get(self, call_id: str) -> dict[str, Any] | None:
        path = self._path(call_id)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read call %s: %s", call_id, exc)
            return None

    def list(self, *, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 500))
        items: list[tuple[float, dict[str, Any]]] = []
        with self._lock:
            for path in self._root.glob("*.json"):
                try:
                    doc = json.loads(path.read_text(encoding="utf-8"))
                    mtime = path.stat().st_mtime
                    items.append((mtime, doc))
                except (OSError, json.JSONDecodeError):
                    continue
        items.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in items[:limit]]

    def delete(self, call_id: str) -> bool:
        path = self._path(call_id)
        with self._lock:
            if not path.is_file():
                return False
            path.unlink()
            return True
