"""FastAPI application factory.

Creates and configures the FastAPI app with exception handlers, lifespan
hooks, and all registered routers.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..core.errors import AnalysisError, BaseAnalysisError, ConfigurationError, TranscriptionError
from .routers import conversation, health, pipeline, scan, text, transcription

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Application lifespan – startup and shutdown hooks."""
    logger.info("Swedish Sentiment API starting up")
    yield
    logger.info("Swedish Sentiment API shutting down")


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


def _error_response(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail})


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured :class:`~fastapi.FastAPI` instance with all routers and
        exception handlers registered.
    """
    app = FastAPI(
        title="Swedish Sentiment API",
        version="0.3.0",
        description=(
            "REST API for Swedish sentiment analysis, ASR transcription, "
            "call-center conversation analysis, and batch processing."
        ),
        lifespan=lifespan,
    )

    # --- Exception handlers --------------------------------------------------

    @app.exception_handler(ConfigurationError)
    async def handle_config_error(request: Request, exc: ConfigurationError) -> JSONResponse:
        logger.warning("Configuration error: %s", exc)
        return _error_response(422, str(exc))

    @app.exception_handler(TranscriptionError)
    async def handle_transcription_error(request: Request, exc: TranscriptionError) -> JSONResponse:
        logger.error("Transcription error: %s", exc)
        return _error_response(500, f"Transcription failed: {exc}")

    @app.exception_handler(AnalysisError)
    async def handle_analysis_error(request: Request, exc: AnalysisError) -> JSONResponse:
        logger.error("Analysis error: %s", exc)
        return _error_response(500, f"Analysis failed: {exc}")

    @app.exception_handler(BaseAnalysisError)
    async def handle_base_error(request: Request, exc: BaseAnalysisError) -> JSONResponse:
        logger.error("Analysis system error: %s", exc)
        return _error_response(500, str(exc))

    # --- Routers -------------------------------------------------------------

    app.include_router(health.router)
    app.include_router(text.router)
    app.include_router(transcription.router)
    app.include_router(conversation.router)
    app.include_router(pipeline.router)
    app.include_router(scan.router)

    return app


# Default application instance (used by uvicorn src.api:app etc.)
app: FastAPI = create_app()
