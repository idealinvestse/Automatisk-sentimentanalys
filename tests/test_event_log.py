"""Tests for launcher event log."""

from __future__ import annotations

from launcher.event_log import EventLog


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