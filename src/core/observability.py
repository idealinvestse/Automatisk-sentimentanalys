"""Observability helpers: phase timers, error boundaries, sampling, job scope."""

from __future__ import annotations

import functools
import logging
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

from .error_helpers import log_and_degrade
from .errors import BaseAnalysisError
from .logging_config import get_logger, log_context
from .status import StatusReporter, get_status_reporter

F = TypeVar("F", bound=Callable[..., Any])


def _parse_sample_env() -> dict[str, int]:
    """Parse SENTIMENT_LOG_SAMPLE=registry=10,asr=5 (keep every Nth DEBUG record)."""
    raw = os.getenv("SENTIMENT_LOG_SAMPLE", "").strip()
    if not raw:
        return {}
    result: dict[str, int] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            prefix, spec = part.split("=", 1)
            if "/" in spec:
                _, denom = spec.split("/", 1)
                result[prefix.strip()] = max(2, int(denom))
            else:
                result[prefix.strip()] = max(2, int(spec))
        else:
            result[part] = 10
    return result


class SamplingFilter(logging.Filter):
    """Drop DEBUG records for chatty loggers (keep every Nth). INFO+ always passes."""

    def __init__(self, sample_config: dict[str, int] | None = None) -> None:
        super().__init__()
        self._config = sample_config if sample_config is not None else _parse_sample_env()
        self._counters: dict[str, int] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno > logging.DEBUG or not self._config:
            return True
        name = record.name
        for prefix, every_n in self._config.items():
            if name.startswith(prefix) or prefix in name:
                key = f"{prefix}:{record.funcName}"
                self._counters[key] = self._counters.get(key, 0) + 1
                return self._counters[key] % every_n == 0
        return True


def apply_component_levels() -> None:
    """Apply SENTIMENT_LOG_COMPONENTS=asr=DEBUG,llm=WARNING to named loggers."""
    raw = os.getenv("SENTIMENT_LOG_COMPONENTS", "").strip()
    if not raw:
        return
    for part in raw.split(","):
        part = part.strip()
        if "=" not in part:
            continue
        comp, level_name = part.split("=", 1)
        level = getattr(logging, level_name.strip().upper(), logging.INFO)
        logging.getLogger(comp.strip()).setLevel(level)


def attach_observability_filters(handler: logging.Handler) -> None:
    """Attach sampling filter and component levels to a handler/root setup."""
    handler.addFilter(SamplingFilter())
    apply_component_levels()


@contextmanager
def job_scope(job_id: str, *, component: str | None = None) -> Iterator[StatusReporter]:
    """Bind job_id (and optional component) for the current scope."""
    with log_context(job_id=job_id, component=component):
        yield get_status_reporter()


@contextmanager
def phase_timer(
    component: str,
    phase: str,
    *,
    job_id: str | None = None,
    **extra: Any,
) -> Iterator[None]:
    """Emit start/complete status with duration; ERROR + re-raise on failure."""
    status = get_status_reporter()
    status.phase(component, phase, f"start {phase}", job_id=job_id, **extra)
    t0 = time.perf_counter()
    failed = False
    try:
        yield
    except Exception as exc:
        failed = True
        status.error(
            component,
            phase,
            f"{phase} misslyckades: {exc}",
            job_id=job_id,
            exc=exc,
            error_code="phase_failed",
        )
        raise
    finally:
        duration_s = round(time.perf_counter() - t0, 3)
        outcome = "avbruten" if failed else "klar"
        status.info(
            component,
            phase,
            f"{outcome} {phase}",
            job_id=job_id,
            duration_s=duration_s,
        )


@contextmanager
def degrading_phase(
    component: str,
    phase: str,
    *,
    results: dict[str, Any],
    result_key: str,
    error_code: str = "analysis_failed",
    job_id: str | None = None,
) -> Iterator[None]:
    """Run a pipeline phase; on failure degrade into results[result_key]."""
    status = get_status_reporter()
    status.phase(component, phase, f"start {phase}", job_id=job_id)
    t0 = time.perf_counter()
    try:
        yield
    except BaseAnalysisError:
        raise
    except Exception as exc:
        payload = log_and_degrade(
            get_logger("pipeline"),
            status,
            component=component,
            phase=phase,
            message=f"{phase} misslyckades",
            exc=exc,
            result_key=result_key,
            error_code=error_code,
        )
        results.update(payload)
    finally:
        status.info(
            component,
            phase,
            f"klar {phase}",
            job_id=job_id,
            duration_s=round(time.perf_counter() - t0, 3),
        )


def with_error_handling(
    component: str,
    phase: str,
    *,
    result_key: str | None = None,
    error_code: str = "analysis_failed",
    raise_on: tuple[type[BaseException], ...] = (BaseAnalysisError,),
) -> Callable[[F], F]:
    """Decorator: timed phase + graceful degradation on unexpected errors."""

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            status = get_status_reporter()
            status.phase(component, phase, f"start {phase}")
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            except raise_on:
                raise
            except BaseException as exc:
                return log_and_degrade(
                    get_logger(fn.__module__),
                    status,
                    component=component,
                    phase=phase,
                    message=f"{phase} misslyckades",
                    exc=exc,
                    result_key=result_key,
                    error_code=error_code,
                )
            finally:
                status.info(
                    component,
                    phase,
                    f"klar {phase}",
                    duration_s=round(time.perf_counter() - t0, 3),
                )

        return wrapper  # type: ignore[return-value]

    return decorator
