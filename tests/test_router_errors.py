"""Tests for shared API router error handling."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from src.api.error_responses import PUBLIC_ERROR_DETAIL
from src.api.router_errors import run_route, run_route_sync
from src.core.errors import AnalysisError, BaseAnalysisError


class TestRunRouteSync:
    def test_returns_success(self) -> None:
        assert run_route_sync("GET /test", lambda: {"ok": True}) == {"ok": True}

    def test_reraises_http_exception(self) -> None:
        def boom() -> None:
            raise HTTPException(status_code=404, detail="missing")

        with pytest.raises(HTTPException) as exc:
            run_route_sync("GET /test", boom)
        assert exc.value.status_code == 404

    def test_reraises_domain_error(self) -> None:
        def boom() -> None:
            raise AnalysisError("analysis failed")

        with pytest.raises(AnalysisError):
            run_route_sync("POST /analyze", boom)

    def test_wraps_generic_exception(self) -> None:
        def boom() -> None:
            raise RuntimeError("unexpected")

        with pytest.raises(HTTPException) as exc:
            run_route_sync("POST /analyze", boom)
        assert exc.value.status_code == 500
        assert exc.value.detail == PUBLIC_ERROR_DETAIL


@pytest.mark.asyncio
class TestRunRoute:
    async def test_returns_success(self) -> None:
        async def fn() -> dict[str, bool]:
            return {"ok": True}

        assert await run_route("GET /test", fn) == {"ok": True}

    async def test_reraises_base_analysis_error(self) -> None:
        async def boom() -> None:
            raise BaseAnalysisError("system")

        with pytest.raises(BaseAnalysisError):
            await run_route("GET /test", boom)

    async def test_wraps_generic_exception(self) -> None:
        async def boom() -> None:
            raise ValueError("bad")

        with pytest.raises(HTTPException) as exc:
            await run_route("GET /test", boom)
        assert exc.value.status_code == 500
