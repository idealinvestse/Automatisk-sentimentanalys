"""Structured API error responses (backward-compatible ``detail`` + extensions)."""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from ..core.errors import (
    AnalysisError,
    BaseAnalysisError,
    ConfigurationError,
    DiarizationError,
    LLMError,
    TranscriptionError,
)

ERROR_CODE_INTERNAL = "internal_error"
ERROR_CODE_VALIDATION = "validation_error"
ERROR_CODE_UNAUTHORIZED = "unauthorized"
ERROR_CODE_RATE_LIMITED = "rate_limit_exceeded"


def request_id_from(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


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
    if isinstance(exc, BaseAnalysisError):
        return exc.error_code
    return ERROR_CODE_INTERNAL


def build_error_content(
    detail: Any,
    *,
    request_id: str | None = None,
    error_code: str | None = None,
) -> dict[str, Any]:
    content: dict[str, Any] = {"detail": detail}
    if request_id:
        content["request_id"] = request_id
    if error_code:
        content["error_code"] = error_code
    return content


def error_response(
    request: Request,
    status_code: int,
    detail: Any,
    *,
    error_code: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=build_error_content(
            detail,
            request_id=request_id_from(request),
            error_code=error_code,
        ),
    )
