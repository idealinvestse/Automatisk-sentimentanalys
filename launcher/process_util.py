"""Cross-platform process existence and service reachability checks."""

from __future__ import annotations

import ctypes
import socket
import sys
import time


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


def resolve_connect_host(bind_host: str) -> str:
    """Map bind-all addresses to a loopback host for client-side probes."""
    normalized = bind_host.strip().lower()
    if normalized in ("0.0.0.0", "::", "[::]"):
        return "127.0.0.1"
    return bind_host


def is_port_open(host: str, port: int, *, timeout: float = 0.5) -> bool:
    """Return True when ``host:port`` accepts a TCP connection."""
    connect_host = resolve_connect_host(host)
    try:
        with socket.create_connection((connect_host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_for_port(
    host: str,
    port: int,
    *,
    timeout_sec: float = 30.0,
    interval_sec: float = 0.25,
) -> bool:
    """Poll until ``host:port`` listens or ``timeout_sec`` elapses."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if is_port_open(host, port, timeout=min(interval_sec, 1.0)):
            return True
        time.sleep(interval_sec)
    return False
