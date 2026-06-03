"""Cross-platform process existence checks."""

from __future__ import annotations

import ctypes
import sys


def is_process_running(pid: int) -> bool:
    """Return True if a process with ``pid`` is still running."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        return _is_running_windows(pid)
    try:
        import os

        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _is_running_windows(pid: int) -> bool:
    query_limited = 0x1000
    still_active = 259
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.OpenProcess(query_limited, False, pid)
    if not handle:
        return False
    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)
