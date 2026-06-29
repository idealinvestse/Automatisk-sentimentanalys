"""Structured logging configuration (PROD-01)."""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from typing import Any, Iterator

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
job_id_var: ContextVar[str | None] = ContextVar("job_id", default=None)
component_var: ContextVar[str | None] = ContextVar("component", default=None)
phase_var: ContextVar[str | None] = ContextVar("phase", default=None)

_CONTEXT_FIELDS = ("request_id", "job_id", "component", "phase")


def _context_payload() -> dict[str, str]:
    payload: dict[str, str] = {}
    for name, var in (
        ("request_id", request_id_var),
        ("job_id", job_id_var),
        ("component", component_var),
        ("phase", phase_var),
    ):
        value = var.get()
        if value:
            payload[name] = value
    return payload


class ContextInjectingFormatter(logging.Formatter):
    """Plain-text formatter with optional observability context fields."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        ctx = _context_payload()
        if not ctx:
            return base
        ctx_str = " ".join(f"{k}={v}" for k, v in ctx.items())
        return f"{base} [{ctx_str}]"


class JSONFormatter(logging.Formatter):
    """Emit one JSON object per log line for log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(_context_payload())
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class _ContextLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that merges bound context into log records."""

    def process(self, msg: str, kwargs: Any) -> tuple[str, Any]:
        extra = dict(kwargs.get("extra") or {})
        for key in _CONTEXT_FIELDS:
            if key not in extra:
                value = self.extra.get(key)
                if value:
                    extra[key] = value
        if extra:
            kwargs["extra"] = extra
        return msg, kwargs


def set_request_id(request_id: str | None) -> None:
    """Bind request_id for the current async/task context."""
    request_id_var.set(request_id)


def set_job_id(job_id: str | None) -> None:
    """Bind job_id for the current async/task context."""
    job_id_var.set(job_id)


def set_component(component: str | None) -> None:
    """Bind component name for the current context."""
    component_var.set(component)


def set_phase(phase: str | None) -> None:
    """Bind phase name for the current context."""
    phase_var.set(phase)


def resolve_log_level() -> int:
    """Resolve log level from environment (LOG_LEVEL, SENTIMENT_LOG_LEVEL, SENTIMENT_DEV)."""
    explicit = os.getenv("LOG_LEVEL") or os.getenv("SENTIMENT_LOG_LEVEL")
    if explicit:
        return getattr(logging, explicit.upper(), logging.INFO)
    if os.getenv("SENTIMENT_DEV", "").lower() in ("1", "true", "yes"):
        return logging.DEBUG
    return logging.INFO


def configure_logging(*, json_logs: bool | None = None, level: int | str | None = None) -> None:
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
            ContextInjectingFormatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
    root.addHandler(handler)
    if level is None:
        root.setLevel(resolve_log_level())
    elif isinstance(level, str):
        root.setLevel(getattr(logging, level.upper(), logging.INFO))
    else:
        root.setLevel(level)


def get_logger(name: str) -> logging.LoggerAdapter:
    """Return a logger that injects current observability context."""
    return _ContextLoggerAdapter(logging.getLogger(name), _context_payload())


@contextmanager
def log_context(
    *,
    request_id: str | None = None,
    job_id: str | None = None,
    component: str | None = None,
    phase: str | None = None,
) -> Iterator[None]:
    """Temporarily bind observability context for the current scope."""
    tokens: list[tuple[ContextVar[str | None], Token[str | None]]] = []
    for var, value in (
        (request_id_var, request_id),
        (job_id_var, job_id),
        (component_var, component),
        (phase_var, phase),
    ):
        if value is not None:
            tokens.append((var, var.set(value)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)
