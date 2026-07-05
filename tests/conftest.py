"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import pytest

# NiceGUI plugin is only needed for NiceGUI dashboard tests (deprecated)
# Load it conditionally to allow API-only tests to run without nicegui installed
try:
    import nicegui  # noqa: F401

    pytest_plugins = ["nicegui.testing.plugin"]
except ImportError:
    pass


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
