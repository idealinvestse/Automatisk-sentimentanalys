"""Generic batch file processor used by the batch and scan endpoints.

Replaces the repeated single-worker / ThreadPoolExecutor pattern that
previously existed in every batch endpoint.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def run_batch(
    files: list[str],
    worker_fn: Callable[[str], T],
    workers: int = 1,
    worker_timeout: float | None = 300.0,
) -> list[tuple[str, T | None, Exception | None]]:
    """Process *files* with *worker_fn*, optionally in parallel.

    Args:
        files: List of file paths to process.
        worker_fn: Callable that accepts a file path and returns a result.
        workers: Number of parallel threads.  When ≤ 1 files are processed
            sequentially in the calling thread.
        worker_timeout: Maximum seconds to wait for each individual future
            when running in parallel (``None`` = no limit).

    Returns:
        List of ``(file_path, result, error)`` tuples in completion order.
        *result* is *None* on error; *error* is *None* on success.
    """
    results: list[tuple[str, T | None, Exception | None]] = []

    if workers <= 1:
        for p in files:
            try:
                results.append((p, worker_fn(p), None))
            except Exception as e:
                logger.error("Worker failed for %s: %s", p, e, exc_info=True)
                results.append((p, None, e))
        return results

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures: dict[Any, str] = {executor.submit(worker_fn, p): p for p in files}
        try:
            for fut in as_completed(futures, timeout=worker_timeout):
                p = futures[fut]
                try:
                    results.append((p, fut.result(), None))
                except Exception as e:
                    logger.error("Worker failed for %s: %s", p, e, exc_info=True)
                    results.append((p, None, e))
        except FuturesTimeoutError:
            # Some futures timed out globally; collect remaining as errors
            completed_paths = {path for path, _, _ in results}
            for fut, p in futures.items():
                if p not in completed_paths:
                    logger.error("Worker timed out for %s", p)
                    if not fut.done():
                        fut.cancel()
                    results.append((p, None, TimeoutError(f"Worker timed out after {worker_timeout}s")))

    return results
