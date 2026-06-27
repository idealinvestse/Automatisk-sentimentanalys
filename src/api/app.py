"""FastAPI application factory.

Creates and configures the FastAPI app with exception handlers, lifespan
hooks, and all registered routers.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..alerting import AlertEngine
from ..alerting_state import AlertingStateManager
from ..caching import AggregateCache
from ..core.errors import (
    AnalysisError,
    BaseAnalysisError,
    ConfigurationError,
    LLMError,
    TranscriptionError,
)
from .dependencies import PUBLIC_ERROR_DETAIL, require_api_key
from .error_responses import (
    ERROR_CODE_INTERNAL,
    ERROR_CODE_RATE_LIMITED,
    ERROR_CODE_UNAUTHORIZED,
    ERROR_CODE_VALIDATION,
    error_code_for,
    error_response,
)
from .middleware_rate_limit import RateLimitMiddleware
from .routers import alerting, conversation, health, pipeline, scan, text, transcription, ws_transcription
from .settings import get_api_settings
from .transcription_events import TranscriptionEventHub
from .transcription_jobs import TranscriptionJobRegistry

logger = logging.getLogger(__name__)


def _init_app_state(application: FastAPI) -> None:
    """Eager init for TestClient when lifespan context is not entered."""
    if hasattr(application.state, "cache"):
        return
    settings = get_api_settings()
    application.state.cache = AggregateCache(
        use_redis=settings.use_redis_cache,
        redis_url=settings.redis_url,
        cache_dir=settings.cache_dir,
    )
    application.state.alert_engine = AlertEngine()
    application.state.alerting_state = AlertingStateManager()
    application.state.alerting_state.sync_to_engine(application.state.alert_engine)
    if not hasattr(application.state, "transcription_events"):
        application.state.transcription_events = TranscriptionEventHub()
    if not hasattr(application.state, "transcription_jobs"):
        application.state.transcription_jobs = TranscriptionJobRegistry()


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Application lifespan – shared cache and alert engine."""
    settings = get_api_settings()
    app.state.cache = AggregateCache(
        use_redis=settings.use_redis_cache,
        redis_url=settings.redis_url,
        cache_dir=settings.cache_dir,
    )
    app.state.alert_engine = AlertEngine()
    app.state.alerting_state = AlertingStateManager()
    app.state.alerting_state.sync_to_engine(app.state.alert_engine)
    hub = TranscriptionEventHub()
    hub.bind_loop(asyncio.get_running_loop())
    app.state.transcription_events = hub
    app.state.transcription_jobs = TranscriptionJobRegistry()
    logger.info("Swedish Sentiment API starting up (auth=%s)", settings.auth_enabled)
    yield
    logger.info("Swedish Sentiment API shutting down")


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured :class:`~fastapi.FastAPI` instance with all routers and
        exception handlers registered.
    """
    settings = get_api_settings()
    app = FastAPI(
        title="Swedish Sentiment API",
        version="0.4.0",
        description=(
            "REST API for Swedish sentiment analysis, ASR transcription, "
            "call-center conversation analysis, and batch processing."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(RequestIdMiddleware)
    if settings.rate_limit_rpm > 0:
        app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.rate_limit_rpm)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # --- Exception handlers --------------------------------------------------

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return error_response(
            request,
            422,
            jsonable_encoder(exc.errors()),
            error_code=ERROR_CODE_VALIDATION,
        )

    @app.exception_handler(HTTPException)
    async def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
        if exc.status_code == 401:
            code = ERROR_CODE_UNAUTHORIZED
        elif exc.status_code == 429:
            code = ERROR_CODE_RATE_LIMITED
        elif exc.status_code == 422 and exc.detail != PUBLIC_ERROR_DETAIL:
            code = ERROR_CODE_VALIDATION
        else:
            code = ERROR_CODE_INTERNAL
        return error_response(request, exc.status_code, exc.detail, error_code=code)

    @app.exception_handler(LLMError)
    async def handle_llm_error(request: Request, exc: LLMError) -> JSONResponse:
        logger.error("LLM error: %s", exc)
        return error_response(request, 502, f"LLM request failed: {exc}", error_code=error_code_for(exc))

    @app.exception_handler(ConfigurationError)
    async def handle_config_error(request: Request, exc: ConfigurationError) -> JSONResponse:
        logger.warning("Configuration error: %s", exc)
        return error_response(request, 422, str(exc), error_code=error_code_for(exc))

    @app.exception_handler(TranscriptionError)
    async def handle_transcription_error(request: Request, exc: TranscriptionError) -> JSONResponse:
        logger.error("Transcription error: %s", exc)
        return error_response(request, 500, f"Transcription failed: {exc}", error_code=error_code_for(exc))

    @app.exception_handler(AnalysisError)
    async def handle_analysis_error(request: Request, exc: AnalysisError) -> JSONResponse:
        logger.error("Analysis error: %s", exc)
        return error_response(request, 500, f"Analysis failed: {exc}", error_code=error_code_for(exc))

    @app.exception_handler(BaseAnalysisError)
    async def handle_base_error(request: Request, exc: BaseAnalysisError) -> JSONResponse:
        logger.error("Analysis system error: %s", exc)
        return error_response(request, 500, str(exc), error_code=error_code_for(exc))

    # --- Routers -------------------------------------------------------------

    _auth = [Depends(require_api_key)]

    app.include_router(health.router)
    app.include_router(text.router, dependencies=_auth)
    app.include_router(transcription.router, dependencies=_auth)
    app.include_router(conversation.router, dependencies=_auth)
    app.include_router(pipeline.router, dependencies=_auth)
    app.include_router(scan.router, dependencies=_auth)
    app.include_router(ws_transcription.router)
    app.include_router(alerting.router, dependencies=_auth)

    _init_app_state(app)
    return app


# Default application instance (used by uvicorn src.api:app etc.)
app: FastAPI = create_app()
