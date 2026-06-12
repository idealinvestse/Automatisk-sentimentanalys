"""Backward-compatible ASGI entrypoint (uvicorn src.api.server:app)."""

from .app import app, create_app

__all__ = ["app", "create_app"]