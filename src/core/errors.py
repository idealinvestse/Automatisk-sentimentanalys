"""Custom exceptions for the Automatic Sentiment Analysis system."""

from __future__ import annotations

from typing import Any


class BaseAnalysisError(Exception):
    """Base exception for all system errors."""

    error_code: str = "analysis_system_error"

    def __init__(
        self,
        message: str | None = None,
        *,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        if message is not None:
            super().__init__(message)
        if error_code is not None:
            self.error_code = error_code
        self.details: dict[str, Any] = dict(details or {})


class TranscriptionError(BaseAnalysisError):
    """Raised when audio transcription fails."""

    error_code = "transcription_failed"


class DiarizationError(BaseAnalysisError):
    """Raised when speaker diarization fails."""

    error_code = "diarization_failed"


class AnalysisError(BaseAnalysisError):
    """Raised when a text analysis step fails."""

    error_code = "analysis_failed"


class ConfigurationError(BaseAnalysisError):
    """Raised when system configuration is invalid or missing."""

    error_code = "configuration_error"


class LLMError(BaseAnalysisError):
    """Raised when an external LLM call (e.g. Mistral via OpenRouter) fails after retries.

    Callers (analyzers, pipeline) should catch and fallback to local heuristics/models.
    Always log external egress for privacy compliance.
    """

    error_code = "llm_request_failed"


class ResourceError(BaseAnalysisError):
    """Raised when a required model, lexicon, or asset is missing."""

    error_code = "resource_unavailable"


class CacheError(BaseAnalysisError):
    """Raised when cache read/write fails."""

    error_code = "cache_error"


class ExternalServiceError(BaseAnalysisError):
    """Raised when an external service (LLM, ASR provider) fails."""

    error_code = "external_service_error"
