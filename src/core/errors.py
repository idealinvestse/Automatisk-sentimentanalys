"""Custom exceptions for the Automatic Sentiment Analysis system."""


class BaseAnalysisError(Exception):
    """Base exception for all system errors."""

    pass


class TranscriptionError(BaseAnalysisError):
    """Raised when audio transcription fails."""

    pass


class DiarizationError(BaseAnalysisError):
    """Raised when speaker diarization fails."""

    pass


class AnalysisError(BaseAnalysisError):
    """Raised when a text analysis step fails."""

    pass


class ConfigurationError(BaseAnalysisError):
    """Raised when system configuration is invalid or missing."""

    pass


class LLMError(BaseAnalysisError):
    """Raised when an external LLM call (e.g. Mistral via OpenRouter) fails after retries.

    Callers (analyzers, pipeline) should catch and fallback to local heuristics/models.
    Always log external egress for privacy compliance.
    """

    pass
