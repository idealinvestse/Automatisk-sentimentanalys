"""Tests for process existence helper."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from launcher.process_util import is_port_open, is_process_running


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


def test_is_port_open_closed() -> None:
    assert is_port_open("127.0.0.1", 59999, timeout=0.2) is False


def test_describe_port_occupant_when_closed() -> None:
    from launcher.process_util import describe_port_occupant

    assert describe_port_occupant("127.0.0.1", 59999) == ""


def test_port_owned_by_pid_tree_false_for_unrelated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from launcher import process_util

    monkeypatch.setattr(process_util, "listening_pids", lambda *a, **k: {19468})
    monkeypatch.setattr(process_util, "process_ancestor_pids", lambda pid: {19468, 1})
    assert process_util.port_owned_by_pid_tree("127.0.0.1", 3000, root_pid=88) is False


def test_port_owned_by_pid_tree_true_for_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from launcher import process_util

    monkeypatch.setattr(process_util, "listening_pids", lambda *a, **k: {500})
    monkeypatch.setattr(process_util, "process_ancestor_pids", lambda pid: {500, 88, 1})
    assert process_util.port_owned_by_pid_tree("127.0.0.1", 3000, root_pid=88) is True
