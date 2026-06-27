"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import pytest

from src.transcription.factory import clear_transcriber_cache

pytest_plugins = ["nicegui.testing.plugin"]


@pytest.fixture(autouse=True)
def _clear_asr_cache() -> None:
    """Ensure the ASR transcriber cache is empty before every test.

    The factory uses ``@lru_cache`` to avoid reloading large Whisper
    models between production requests, but that state leaks across
    tests unless we clear it.
    """
    clear_transcriber_cache()
