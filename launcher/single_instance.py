"""Ensure only one launcher GUI instance runs per user session."""

from __future__ import annotations

import sys
from types import TracebackType


class SingleInstanceLock:
    """Windows named-mutex lock; no-op elsewhere.

    Keep an instance alive for the process lifetime so the mutex is held.
    """

    def __init__(self, name: str = "Local\\SentimentanalysLauncher") -> None:
        self._name = name
        self._handle: int | None = None
        self.acquired = False

    def acquire(self) -> bool:
        if sys.platform != "win32":
            self.acquired = True
            return True
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        error_already_exists = 183
        handle = kernel32.CreateMutexW(None, False, self._name)
        if not handle:
            self.acquired = False
            return False
        self._handle = int(handle)
        if kernel32.GetLastError() == error_already_exists:
            kernel32.CloseHandle(handle)
            self._handle = None
            self.acquired = False
            return False
        self.acquired = True
        return True

    def release(self) -> None:
        if self._handle is None or sys.platform != "win32":
            self._handle = None
            self.acquired = False
            return
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        kernel32.CloseHandle(self._handle)
        self._handle = None
        self.acquired = False

    def __enter__(self) -> SingleInstanceLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()


def try_acquire_launcher_lock() -> SingleInstanceLock | None:
    """Return a held lock, or None when another launcher is already running."""
    lock = SingleInstanceLock()
    if lock.acquire():
        return lock
    return None
