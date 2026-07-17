from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")

_OOM_MARKERS = (
    "out of memory",
    "cuda out of memory",
    "cudnn_status_alloc_failed",
    "hip out of memory",
)


@dataclass(frozen=True)
class OomFallbackResult:
    value: object
    model_used: str
    fell_back: bool


def is_cuda_oom_error(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    if "outofmemory" in name:
        return True
    msg = str(exc).lower()
    return any(marker in msg for marker in _OOM_MARKERS)


def transcribe_with_oom_fallback(
    *,
    primary_model: str = "kb-whisper-large",
    fallback_model: str = "kb-whisper-medium",
    allow_fallback: bool = True,
    on_fallback: Callable[[str, str, BaseException], None] | None = None,
    transcribe_fn: Callable[[str], T],
) -> OomFallbackResult:
    try:
        value = transcribe_fn(primary_model)
        return OomFallbackResult(value=value, model_used=primary_model, fell_back=False)
    except Exception as exc:
        if not allow_fallback or not is_cuda_oom_error(exc):
            raise
        if fallback_model == primary_model:
            raise
        if on_fallback is not None:
            on_fallback(primary_model, fallback_model, exc)
        else:
            logger.warning(
                "CUDA OOM with model %s; retrying once with %s",
                primary_model,
                fallback_model,
            )
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        value = transcribe_fn(fallback_model)
        return OomFallbackResult(value=value, model_used=fallback_model, fell_back=True)
