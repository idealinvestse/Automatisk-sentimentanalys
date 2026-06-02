"""API package – exposes the FastAPI app instance and factory function."""

from .app import app, create_app

__all__ = ["app", "create_app"]
