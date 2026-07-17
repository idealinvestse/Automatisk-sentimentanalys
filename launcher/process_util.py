"""Cross-platform process existence and service reachability checks."""

from __future__ import annotations

import ctypes
import re
import socket
import subprocess
import sys
import time
from pathlib import Path


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


_NETSTAT_LISTEN_RE = re.compile(
    r"^\s*TCP\s+(\S+):(\d+)\s+\S+\s+LISTENING\s+(\d+)\s*$",
    re.IGNORECASE,
)


def listening_pids(host: str, port: int) -> set[int]:
    """Return PIDs with a TCP LISTENING socket on ``host:port`` (best-effort)."""
    connect_host = resolve_connect_host(host)
    want_hosts = {connect_host.lower(), "0.0.0.0", "*", "::", "[::]"}
    if connect_host == "127.0.0.1":
        want_hosts.add("127.0.0.1")
    pids: set[int] = set()
    if sys.platform == "win32":
        try:
            out = subprocess.check_output(
                ["netstat", "-ano", "-p", "tcp"],
                text=True,
                errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW
                if hasattr(subprocess, "CREATE_NO_WINDOW")
                else 0,
            )
        except (OSError, subprocess.CalledProcessError):
            return pids
        for line in out.splitlines():
            match = _NETSTAT_LISTEN_RE.match(line)
            if not match:
                continue
            addr, port_s, pid_s = match.groups()
            if int(port_s) != port:
                continue
            addr_norm = addr.strip("[]").lower()
            if addr_norm not in want_hosts and addr_norm not in {"0.0.0.0", "::"}:
                # Allow any local bind that matches the requested port when probing loopback.
                if connect_host == "127.0.0.1" and addr_norm in {"127.0.0.1", "0.0.0.0"}:
                    pass
                else:
                    continue
            try:
                pid = int(pid_s)
            except ValueError:
                continue
            if pid > 0:
                pids.add(pid)
        return pids

    # Linux / macOS: ss or lsof best-effort
    try:
        out = subprocess.check_output(
            ["ss", "-ltnp"],
            text=True,
            errors="replace",
        )
        for line in out.splitlines():
            if f":{port}" not in line or "users:" not in line:
                continue
            for pid_s in re.findall(r"pid=(\d+)", line):
                pids.add(int(pid_s))
    except (OSError, subprocess.CalledProcessError):
        pass
    return pids


def process_parent_pid(pid: int) -> int | None:
    """Return the parent PID for ``pid``, or None if unknown."""
    if pid <= 0:
        return None
    if sys.platform == "win32":
        return _parent_pid_windows(pid)
    try:
        status = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        # Field 4 is ppid after comm in parentheses.
        close = status.rfind(")")
        if close < 0:
            return None
        fields = status[close + 2 :].split()
        return int(fields[1])
    except (OSError, IndexError, ValueError):
        return None


def _parent_pid_windows(pid: int) -> int | None:
    class PROCESSENTRY32(ctypes.Structure):
        _fields_ = [
            ("dwSize", ctypes.c_ulong),
            ("cntUsage", ctypes.c_ulong),
            ("th32ProcessID", ctypes.c_ulong),
            ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
            ("th32ModuleID", ctypes.c_ulong),
            ("cntThreads", ctypes.c_ulong),
            ("th32ParentProcessID", ctypes.c_ulong),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", ctypes.c_ulong),
            ("szExeFile", ctypes.c_wchar * 260),
        ]

    th32cs_snapprocess = 0x00000002
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    snapshot = kernel32.CreateToolhelp32Snapshot(th32cs_snapprocess, 0)
    if snapshot == ctypes.c_void_p(-1).value or snapshot == -1:
        return None
    try:
        entry = PROCESSENTRY32()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32)
        if not kernel32.Process32FirstW(snapshot, ctypes.byref(entry)):
            return None
        while True:
            if entry.th32ProcessID == pid:
                return int(entry.th32ParentProcessID)
            if not kernel32.Process32NextW(snapshot, ctypes.byref(entry)):
                break
    finally:
        kernel32.CloseHandle(snapshot)
    return None


def process_ancestor_pids(pid: int, *, max_depth: int = 32) -> set[int]:
    """Return ``pid`` plus its ancestors (best-effort, cycle-safe)."""
    seen: set[int] = set()
    current = pid
    for _ in range(max_depth):
        if current <= 0 or current in seen:
            break
        seen.add(current)
        parent = process_parent_pid(current)
        if parent is None or parent == current:
            break
        current = parent
    return seen


def port_owned_by_pid_tree(host: str, port: int, root_pid: int) -> bool:
    """True if a listener on ``host:port`` is ``root_pid`` or one of its descendants."""
    if root_pid <= 0:
        return False
    listeners = listening_pids(host, port)
    if not listeners:
        return False
    if root_pid in listeners:
        return True
    for listener in listeners:
        if root_pid in process_ancestor_pids(listener):
            return True
    return False


def process_display_name(pid: int) -> str:
    """Best-effort executable name for ``pid``."""
    if pid <= 0:
        return f"pid {pid}"
    if sys.platform == "win32":
        try:
            out = subprocess.check_output(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                text=True,
                errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW
                if hasattr(subprocess, "CREATE_NO_WINDOW")
                else 0,
            ).strip()
            if out and out.lower() not in {"info: no tasks are running which match the specified criteria.", ""}:
                # "name.exe","pid","session","session#","mem"
                name = out.split(",")[0].strip().strip('"')
                if name:
                    return name
        except (OSError, subprocess.CalledProcessError):
            pass
    else:
        try:
            return Path(f"/proc/{pid}/comm").read_text(encoding="utf-8").strip() or f"pid {pid}"
        except OSError:
            pass
    return f"pid {pid}"


def describe_port_occupant(host: str, port: int) -> str:
    """Human-readable summary of who listens on ``host:port``, or empty if free."""
    pids = listening_pids(host, port)
    if not pids:
        if is_port_open(host, port, timeout=0.2):
            return "unknown process (port accepts connections)"
        return ""
    parts = [f"{process_display_name(pid)} (pid {pid})" for pid in sorted(pids)]
    return ", ".join(parts)
