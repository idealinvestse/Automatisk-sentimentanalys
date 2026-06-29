"""Tests for shared API router error handling."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from src.api.error_responses import PUBLIC_ERROR_DETAIL
from src.api.router_errors import run_route, run_route_sync
from src.core.errors import AnalysisError, BaseAnalysisError


@pytest.mark.asyncio
async def test_run_route_success() -> None:
    async def ok():
        return {"ok": True}

    result = await run_route("GET /test", ok)
    assert result == {"ok": True}


@pytest.mark.asyncio
async def test_run_route_reraises_http_exception() -> None:
    async def boom():
        raise HTTPException(status_code=404, detail="missing")

    with pytest.raises(HTTPException) as exc:
        await run_route("GET /test", boom)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_run_route_reraises_base_analysis_error() -> None:
    async def boom():
        raise AnalysisError("analysis failed")

    with pytest.raises(BaseAnalysisError):
        await run_route("GET /test", boom)


@pytest.mark.asyncio
async def test_run_route_wraps_generic_exception() -> None:
    async def boom():
        raise RuntimeError("unexpected")

    with pytest.raises(HTTPException) as exc:
        await run_route("POST /test", boom)
    assert exc.value.status_code == 500
    assert exc.value.detail == PUBLIC_ERROR_DETAIL


def test_run_route_sync_success() -> None:
    def ok():
        return 42

    assert run_route_sync("GET /sync", ok) == 42


def test_run_route_sync_reraises_http_exception() -> None:
    def boom():
        raise HTTPException(status_code=403, detail="forbidden")

    with pytest.raises(HTTPException) as exc:
        run_route_sync("GET /sync", boom)
    assert exc.value.status_code == 403


def test_run_route_sync_reraises_base_analysis_error() -> None:
    def boom():
        raise AnalysisError("bad step")

    with pytest.raises(BaseAnalysisError):
        run_route_sync("GET /sync", boom)


def test_run_route_sync_wraps_generic_exception() -> None:
    def boom():
        raise ValueError("bad input")

    with pytest.raises(HTTPException) as exc:
        run_route_sync("GET /sync", boom)
    assert exc.value.status_code == 500
    assert exc.value.detail == PUBLIC_ERROR_DETAIL
