"""Health check router."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health() -> dict[str, str]:
    """Simple health check endpoint.

    Returns:
        ``{"status": "ok"}`` when the service is running.
    """
    return {"status": "ok"}
