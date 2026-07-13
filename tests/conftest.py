"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clear_asr_cache() -> None:
    """Ensure the ASR transcriber cache is empty before every test.

    The factory uses ``@lru_cache`` to avoid reloading large Whisper
    models between production requests, but that state leaks across
    tests unless we clear it.

    This import is lazy to allow tests to run without torch installed
    (graceful degradation for API-only tests).
    """
    try:
        from src.transcription.factory import clear_transcriber_cache

        clear_transcriber_cache()
    except ImportError:
        # torch not installed, skip cache clearing (ASR tests will fail gracefully)
        pass
