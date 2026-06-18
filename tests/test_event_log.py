"""Tests for launcher event log."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from launcher.event_log import EventLog, _parse_log_line
from launcher.pid_store import launcher_activity_log_path
from src.install.config_schema import UserConfig


def test_event_log_ring_buffer() -> None:
    log = EventLog(max_entries=3)
    log.info("a")
    log.info("b")
    log.info("c")
    log.info("d")
    entries = log.entries()
    assert len(entries) == 3
    assert entries[0].message == "b"


def test_event_log_format_and_poll() -> None:
    log = EventLog()
    log.phase("api.start", "Starting")
    polled = log.poll_queue()
    assert len(polled) == 1
    line = polled[0].format_line()
    assert "PHASE" in line
    assert "[api.start]" in line
    assert log.poll_queue() == []


def test_event_log_clear() -> None:
    log = EventLog()
    log.info("x")
    log.clear()
    assert log.entries() == []
    assert log.poll_queue() == []


def test_append_writes_to_file(tmp_path: Path) -> None:
    log_file = tmp_path / "launcher_activity.log"
    log = EventLog(log_path=log_file, load_max_lines=0)
    log.info("persistent line", phase="test")
    assert log_file.is_file()
    content = log_file.read_text(encoding="utf-8")
    assert "persistent line" in content
    assert "[test]" in content
    assert content.startswith("20")  # full date prefix


def test_load_recent_reads_tail(tmp_path: Path) -> None:
    log_file = tmp_path / "launcher_activity.log"
    lines = []
    for i in range(600):
        lines.append(f"2026-06-18 10:00:00.{i % 1000:03d} INFO  msg-{i}")
    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    log = EventLog(max_entries=500, log_path=None)
    loaded = log.load_recent(log_file, max_lines=500)
    assert loaded == 500
    assert len(log.entries()) == 500
    assert log.entries()[0].message == "msg-100"
    assert log.entries()[-1].message == "msg-599"


def test_clear_does_not_truncate_file(tmp_path: Path) -> None:
    log_file = tmp_path / "launcher_activity.log"
    log = EventLog(log_path=log_file, load_max_lines=0)
    log.info("keep me")
    log.clear()
    assert log.entries() == []
    assert "keep me" in log_file.read_text(encoding="utf-8")


def test_thread_safe_append(tmp_path: Path) -> None:
    log_file = tmp_path / "launcher_activity.log"
    log = EventLog(log_path=log_file, load_max_lines=0)

    def worker(n: int) -> None:
        for i in range(20):
            log.info(f"t{n}-{i}")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    lines = [ln for ln in log_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 100
    for line in lines:
        assert _parse_log_line(line) is not None


def test_parse_log_line_short_timestamp() -> None:
    event = _parse_log_line("12:34:56.789 INFO  hello")
    assert event is not None
    assert event.level == "INFO"
    assert event.message == "hello"


def test_launcher_activity_log_path(tmp_path: Path) -> None:
    cfg = UserConfig(paths={"app_root": str(tmp_path)}, portable_mode=True)
    path = launcher_activity_log_path(cfg)
    assert path.name == "launcher_activity.log"
    assert path.parent == tmp_path / "user_data" / "logs"
    assert path.parent.is_dir()