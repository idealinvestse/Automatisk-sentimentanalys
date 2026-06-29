"""Helpers for consistent error logging and graceful degradation."""

from __future__ import annotations

import logging
from typing import Any

from .status import StatusReporter, get_status_reporter


def log_and_degrade(
    logger: logging.Logger | logging.LoggerAdapter,
    status: StatusReporter | None,
    *,
    component: str,
    phase: str,
    message: str,
    exc: BaseException,
    result_key: str | None = None,
    error_code: str = "analysis_failed",
    fallback: bool = True,
) -> dict[str, Any]:
    """Log an error, emit status, and return a partial result dict for graceful degradation."""
    full_message = f"{message}: {exc}"
    logger.warning(full_message, exc_info=True)
    reporter = status or get_status_reporter()
    reporter.warn(component, phase, full_message, error_code=error_code)
    payload: dict[str, Any] = {
        "error": str(exc),
        "error_code": error_code,
        "fallback": fallback,
    }
    if result_key:
        return {result_key: payload}
    return payload
