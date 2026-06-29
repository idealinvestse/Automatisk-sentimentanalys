"""Structured API error responses (backward-compatible ``detail`` + extensions)."""

from __future__ import annotations

import os
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from ..core.errors import (
    AnalysisError,
    BaseAnalysisError,
    CacheError,
    ConfigurationError,
    DiarizationError,
    ExternalServiceError,
    LLMError,
    ResourceError,
    TranscriptionError,
)

ERROR_CODE_INTERNAL = "internal_error"
ERROR_CODE_VALIDATION = "validation_error"
ERROR_CODE_UNAUTHORIZED = "unauthorized"
ERROR_CODE_RATE_LIMITED = "rate_limit_exceeded"

PUBLIC_ERROR_DETAIL = "An internal error occurred. Please try again later."
CONFIGURATION_ERROR_DETAIL = "Invalid configuration."
LLM_ERROR_DETAIL = "LLM request failed. Please try again later."
TRANSCRIPTION_ERROR_DETAIL = "Transcription failed. Please try again later."
ANALYSIS_ERROR_DETAIL = "Analysis failed. Please try again later."


def request_id_from(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _dev_mode() -> bool:
    return os.getenv("SENTIMENT_DEV", "").lower() in ("1", "true", "yes")


def public_detail(exc: BaseException, *, dev_prefix: str | None = None, public: str = PUBLIC_ERROR_DETAIL) -> str:
    """Return a safe client-facing message; include exception text only in dev mode."""
    if _dev_mode():
        if dev_prefix:
            return f"{dev_prefix}: {exc}"
        return str(exc)
    return public


def error_code_for(exc: BaseException) -> str:
    if isinstance(exc, LLMError):
        return exc.error_code
    if isinstance(exc, ConfigurationError):
        return exc.error_code
    if isinstance(exc, TranscriptionError):
        return exc.error_code
    if isinstance(exc, DiarizationError):
        return exc.error_code
    if isinstance(exc, AnalysisError):
        return exc.error_code
    if isinstance(exc, ResourceError):
        return exc.error_code
    if isinstance(exc, CacheError):
        return exc.error_code
    if isinstance(exc, ExternalServiceError):
        return exc.error_code
    if isinstance(exc, BaseAnalysisError):
        return exc.error_code
    return ERROR_CODE_INTERNAL


def build_error_content(
    detail: Any,
    *,
    request_id: str | None = None,
    error_code: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    content: dict[str, Any] = {"detail": detail}
    if request_id:
        content["request_id"] = request_id
    if error_code:
        content["error_code"] = error_code
    if details and _dev_mode():
        content["details"] = details
    return content


def error_response(
    request: Request,
    status_code: int,
    detail: Any,
    *,
    error_code: str | None = None,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=build_error_content(
            detail,
            request_id=request_id_from(request),
            error_code=error_code,
            details=details,
        ),
    )
