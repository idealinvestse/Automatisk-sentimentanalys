"""Tests for process existence helper."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from launcher.process_util import is_process_running


@pytest.mark.skipif(sys.platform != "win32", reason="Windows OpenProcess check")
def test_is_process_running_current_process() -> None:
    assert is_process_running(os.getpid()) is True


@pytest.mark.skipif(sys.platform != "win32", reason="Windows OpenProcess check")
def test_is_process_running_dead_pid() -> None:
    proc = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    proc.wait(timeout=30)
    assert is_process_running(proc.pid) is False


def test_is_process_running_invalid_pid() -> None:
    assert is_process_running(-1) is False
    assert is_process_running(0) is False
