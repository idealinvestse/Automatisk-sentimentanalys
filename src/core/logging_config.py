"""Structured logging configuration (PROD-01)."""

from __future__ import annotations

import json
import logging
import os
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


class JSONFormatter(logging.Formatter):
    """Emit one JSON object per log line for log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        rid = request_id_var.get()
        if rid:
            payload["request_id"] = rid
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def set_request_id(request_id: str | None) -> None:
    """Bind request_id for the current async/task context."""
    request_id_var.set(request_id)


def configure_logging(*, json_logs: bool | None = None) -> None:
    """Configure root logging. Uses JSON when ``SENTIMENT_JSON_LOGS=1`` or *json_logs*."""
    if json_logs is None:
        json_logs = os.getenv("SENTIMENT_JSON_LOGS", "").lower() in ("1", "true", "yes")

    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler()
    if json_logs:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
    root.addHandler(handler)
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    root.setLevel(getattr(logging, level, logging.INFO))
