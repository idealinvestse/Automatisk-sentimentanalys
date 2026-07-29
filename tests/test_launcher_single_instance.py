"""Tests for launcher single-instance lock."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from launcher.single_instance import SingleInstanceLock, try_acquire_launcher_lock


@pytest.mark.skipif(sys.platform == "win32", reason="non-Windows path")
def test_lock_is_noop_outside_windows() -> None:
    lock = SingleInstanceLock()
    assert lock.acquire() is True
    lock.release()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows mutex")
def test_second_lock_fails_while_first_held() -> None:
    first = SingleInstanceLock(name="Local\\SentimentanalysLauncherTestMutex")
    second = SingleInstanceLock(name="Local\\SentimentanalysLauncherTestMutex")
    assert first.acquire() is True
    try:
        assert second.acquire() is False
    finally:
        first.release()
    assert second.acquire() is True
    second.release()


def test_try_acquire_launcher_lock_returns_none_when_busy() -> None:
    with patch("launcher.single_instance.SingleInstanceLock") as cls:
        inst = MagicMock()
        inst.acquire.return_value = False
        cls.return_value = inst
        assert try_acquire_launcher_lock() is None
