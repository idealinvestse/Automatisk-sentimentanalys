"""Server-side call history (analyzed reports)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from ..call_store import CallStore
from ..settings import get_api_settings

router = APIRouter(prefix="/calls", tags=["Calls"])


class CallUpsertRequest(BaseModel):
    """Store or update an analyzed call."""

    id: str = Field(..., min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._-]+$")
    transcript: dict[str, Any] = Field(default_factory=dict)
    report: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None


class CallListResponse(BaseModel):
    calls: list[dict[str, Any]]
    count: int


def _store(request: Request) -> CallStore:
    store = getattr(request.app.state, "call_store", None)
    if store is None:
        settings = get_api_settings()
        store = CallStore(settings.state_dir)
        request.app.state.call_store = store
    return store


@router.get("", response_model=CallListResponse)
async def list_calls(
    request: Request,
    limit: int = Query(50, ge=1, le=500),
) -> CallListResponse:
    """List recently saved analyzed calls (newest first)."""
    calls = _store(request).list(limit=limit)
    return CallListResponse(calls=calls, count=len(calls))


@router.get("/{call_id}")
async def get_call(call_id: str, request: Request) -> dict[str, Any]:
    """Fetch a single saved call by id."""
    doc = _store(request).get(call_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Call not found")
    return doc


@router.put("/{call_id}")
async def upsert_call(call_id: str, body: CallUpsertRequest, request: Request) -> dict[str, Any]:
    """Create or update a saved call (id in path must match body.id)."""
    if body.id != call_id:
        raise HTTPException(status_code=422, detail="Path id must match body.id")
    try:
        return _store(request).save(
            call_id,
            {
                "transcript": body.transcript,
                "report": body.report,
                "meta": body.meta,
                "created_at": body.created_at,
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("")
async def create_call(body: CallUpsertRequest, request: Request) -> dict[str, Any]:
    """Create or update a saved call."""
    try:
        return _store(request).save(
            body.id,
            {
                "transcript": body.transcript,
                "report": body.report,
                "meta": body.meta,
                "created_at": body.created_at,
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.delete("/{call_id}")
async def delete_call(call_id: str, request: Request) -> dict[str, Any]:
    """Delete a saved call."""
    ok = _store(request).delete(call_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Call not found")
    return {"id": call_id, "deleted": True}
