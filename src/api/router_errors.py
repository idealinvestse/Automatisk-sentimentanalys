"""Shared router error handling — re-raise domain errors, sanitize generic failures."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

from fastapi import HTTPException

from ..core.errors import BaseAnalysisError
from .dependencies import PUBLIC_ERROR_DETAIL

logger = logging.getLogger(__name__)
_T = TypeVar("_T")


async def run_route(endpoint: str, fn: Callable[[], Awaitable[_T]]) -> _T:
    try:
        return await fn()  # type: ignore[no-any-return]
    except HTTPException:
        raise
    except BaseAnalysisError:
        raise
    except Exception as e:
        logger.error("%s failed: %s", endpoint, e, exc_info=True)
        raise HTTPException(status_code=500, detail=PUBLIC_ERROR_DETAIL) from e


def run_route_sync(endpoint: str, fn: Callable[[], _T]) -> _T:
    """Sync variant for non-async router handlers."""
    try:
        return fn()
    except HTTPException:
        raise
    except BaseAnalysisError:
        raise
    except Exception as e:
        logger.error("%s failed: %s", endpoint, e, exc_info=True)
        raise HTTPException(status_code=500, detail=PUBLIC_ERROR_DETAIL) from e
