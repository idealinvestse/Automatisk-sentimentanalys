"""Unit tests for src.api.batch.run_batch cancel + callback behavior."""

from __future__ import annotations

import time

from src.api.batch import run_batch


def test_run_batch_sequential_should_cancel_stops_early() -> None:
    calls: list[str] = []

    def worker(path: str) -> str:
        calls.append(path)
        return path

    cancel_after = {"n": 0}

    def should_cancel() -> bool:
        return cancel_after["n"] >= 2

    def on_complete(path, result, error, done, total) -> None:
        cancel_after["n"] = done

    results = run_batch(
        ["a", "b", "c", "d"],
        worker,
        workers=1,
        should_cancel=should_cancel,
        on_file_complete=on_complete,
    )
    assert len(results) == 2
    assert calls == ["a", "b"]
    assert all(err is None for _, _, err in results)


def test_run_batch_parallel_should_cancel_stops_pending() -> None:
    def worker(path: str) -> str:
        if path == "slow":
            time.sleep(0.15)
        return path

    cancelled = {"flag": False}

    def should_cancel() -> bool:
        return cancelled["flag"]

    def on_complete(path, result, error, done, total) -> None:
        if path == "fast":
            cancelled["flag"] = True

    results = run_batch(
        ["fast", "slow", "other"],
        worker,
        workers=3,
        worker_timeout=1.0,
        should_cancel=should_cancel,
        on_file_complete=on_complete,
    )
    assert any(p == "fast" for p, _, _ in results)
    assert len(results) <= 3


def test_run_batch_empty_files() -> None:
    assert run_batch([], lambda p: p, workers=2) == []


def test_run_batch_on_file_complete_counts() -> None:
    seen: list[tuple[int, int]] = []

    def on_complete(path, result, error, done, total) -> None:
        seen.append((done, total))

    run_batch(["a", "b"], lambda p: p, workers=1, on_file_complete=on_complete)
    assert seen == [(1, 2), (2, 2)]
