"""File-backed store for analyzed call reports (server-side call history)."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def new_call_id() -> str:
    """Issue a server-owned stable call identifier."""
    return uuid.uuid4().hex


def call_idempotency_key(
    customer_id: str | None,
    source: str,
    config_fingerprint: str | None,
) -> str | None:
    """Idempotency key: customer + source filename + frozen config fingerprint.

    Returns ``None`` when there is no customer and no source name — otherwise
    every anonymous ``/analyze_pipeline`` call would collide on the same key.
    """
    cid = (customer_id or "").strip().lower()
    name = Path(source or "").name.strip()
    if not cid and not name:
        return None
    raw = f"{cid or '-'}|{name or '-'}|{(config_fingerprint or '-').strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


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
        existing = self.get(call_id)
        created_at = payload.get("created_at") or (existing or {}).get("created_at") or _utc_now()
        existing_meta = dict((existing or {}).get("meta") or {})
        incoming_meta = payload.get("meta")
        meta = {**existing_meta, **incoming_meta} if incoming_meta is not None else existing_meta
        existing_prov = dict((existing or {}).get("provenance") or {})
        incoming_prov = payload.get("provenance")
        provenance = (
            {**existing_prov, **incoming_prov} if incoming_prov is not None else existing_prov
        )
        doc = {
            "id": call_id,
            "created_at": created_at,
            "updated_at": _utc_now(),
            "customer_id": payload.get("customer_id")
            if payload.get("customer_id") is not None
            else (existing or {}).get("customer_id"),
            "status": payload.get("status") or (existing or {}).get("status") or "completed",
            "idempotency_key": payload.get("idempotency_key")
            or (existing or {}).get("idempotency_key"),
            "transcript": payload.get("transcript")
            if payload.get("transcript") is not None
            else (existing or {}).get("transcript") or {},
            "report": payload.get("report")
            if payload.get("report") is not None
            else (existing or {}).get("report") or {},
            "meta": meta,
            "provenance": provenance,
        }
        with self._lock:
            path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        return doc

    def get(self, call_id: str) -> dict[str, Any] | None:
        path = self._path(call_id)
        if not path.is_file():
            return None
        try:
            loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            return loaded
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

    def find_by_idempotency(self, key: str | None) -> dict[str, Any] | None:
        """Return the stored call matching an idempotency key.

        Scans every record. ``list(limit=500)`` is a UI window and must not
        hide older keys.
        """
        if not key:
            return None
        with self._lock:
            for path in self._root.glob("*.json"):
                try:
                    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if loaded.get("idempotency_key") == key:
                    return loaded
        return None

    def delete(self, call_id: str) -> bool:
        path = self._path(call_id)
        with self._lock:
            if not path.is_file():
                return False
            path.unlink()
            return True
