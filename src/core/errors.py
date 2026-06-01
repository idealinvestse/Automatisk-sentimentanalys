"""Custom exceptions for the Automatic Sentiment Analysis system."""


class SystemError(Exception):
    """Base exception for all system errors."""

    pass


class TranscriptionError(SystemError):
    """Raised when audio transcription fails."""

    pass


class DiarizationError(SystemError):
    """Raised when speaker diarization fails."""

    pass


class AnalysisError(SystemError):
    """Raised when a text analysis step fails."""

    pass


class ConfigurationError(SystemError):
    """Raised when system configuration is invalid or missing."""

    pass
