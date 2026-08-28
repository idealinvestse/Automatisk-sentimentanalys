"""FastAPI application factory.

Creates and configures the FastAPI app with exception handlers, lifespan
hooks, and all registered routers.
"""

from __future__ import annotations

import asyncio
import logging
import time
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
from ..core.logging_config import configure_logging, set_job_id, set_request_id
from ..core.status import get_status_reporter
from ..core.tracing import init_tracing
from ..core.version import get_package_version
from .call_store import CallStore
from .dependencies import require_api_key
from .error_responses import (
    ANALYSIS_ERROR_DETAIL,
    CONFIGURATION_ERROR_DETAIL,
    ERROR_CODE_CONFLICT,
    ERROR_CODE_FORBIDDEN,
    ERROR_CODE_INTERNAL,
    ERROR_CODE_NOT_FOUND,
    ERROR_CODE_PAYLOAD_TOO_LARGE,
    ERROR_CODE_RATE_LIMITED,
    ERROR_CODE_UNAUTHORIZED,
    ERROR_CODE_VALIDATION,
    LLM_ERROR_DETAIL,
    PUBLIC_ERROR_DETAIL,
    TRANSCRIPTION_ERROR_DETAIL,
    error_code_for,
    error_response,
    public_detail,
)
from .metrics import init_app_info, record_http_request
from .middleware_rate_limit import RateLimitMiddleware
from .routers import (
    alerting,
    calls,
    conversation,
    edge,
    health,
    llm_profiles,
    pipeline,
    scan,
    status,
    text,
    transcription,
    ws_transcription,
)
from .settings import get_api_settings, validate_production_settings
from .transcription_events import JOB_HEADER, TranscriptionEventHub
from .transcription_job_store import create_job_store
from .transcription_jobs import TranscriptionJobRegistry
from .ws_tickets import TicketStore

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
        redis_client = getattr(application.state.cache, "redis_client", None)
        application.state.transcription_events = TranscriptionEventHub(redis_client=redis_client)
    if not hasattr(application.state, "transcription_jobs"):
        application.state.transcription_jobs = TranscriptionJobRegistry(
            store=create_job_store(
                state_dir=settings.state_dir,
                use_redis=settings.use_redis_cache,
                redis_client=getattr(application.state.cache, "redis_client", None),
            )
        )
    if not hasattr(application.state, "ws_tickets"):
        application.state.ws_tickets = TicketStore(
            redis_client=application.state.cache.redis_client,
        )
    if not hasattr(application.state, "call_store"):
        application.state.call_store = CallStore(settings.state_dir)


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        job_id = request.headers.get(JOB_HEADER) or request_id
        request.state.request_id = request_id
        set_request_id(request_id)
        set_job_id(job_id)
        started = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - started
        route = request.scope.get("route")
        metric_path = route.path if route is not None else request.url.path
        record_http_request(request.method, metric_path, response.status_code, duration)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Application lifespan – shared cache and alert engine."""
    settings = get_api_settings()
    validate_production_settings(settings)
    configure_logging()
    init_tracing(service_name="sentiment-api")
    app.state.cache = AggregateCache(
        use_redis=settings.use_redis_cache,
        redis_url=settings.redis_url,
        cache_dir=settings.cache_dir,
    )
    app.state.alert_engine = AlertEngine()
    app.state.alerting_state = AlertingStateManager()
    app.state.alerting_state.sync_to_engine(app.state.alert_engine)
    hub = TranscriptionEventHub(redis_client=app.state.cache.redis_client)
    hub.bind_loop(asyncio.get_running_loop())
    await hub.start_redis_listener()
    app.state.transcription_events = hub
    app.state.transcription_jobs = TranscriptionJobRegistry(
        store=create_job_store(
            state_dir=settings.state_dir,
            use_redis=settings.use_redis_cache,
            redis_client=app.state.cache.redis_client,
        )
    )
    app.state.ws_tickets = TicketStore(redis_client=app.state.cache.redis_client)
    app.state.call_store = CallStore(settings.state_dir)
    init_app_info(version=get_package_version())

    # Startup upload retention cleanup (also runs on each upload)
    if settings.media_root:
        try:
            from pathlib import Path

            from .routers.transcription import _cleanup_old_uploads

            upload_dir = Path(settings.media_root) / "uploads"
            _cleanup_old_uploads(upload_dir, settings.upload_retention_days)
        except Exception as exc:
            logger.warning("Startup upload cleanup skipped: %s", exc)

    if settings.use_redis_cache and hub.backend != "redis":
        logger.warning(
            "API_USE_REDIS_CACHE=true but transcription event hub backend=%s — "
            "live WS will not fan out across uvicorn workers",
            hub.backend,
        )
    if settings.production and not settings.use_redis_cache:
        logger.warning(
            "API_PRODUCTION without API_USE_REDIS_CACHE — use a single uvicorn worker "
            "or enable Redis for jobs/tickets/WS events"
        )

    reporter = get_status_reporter()
    reporter.phase("api", "startup", "Swedish Sentiment API started", auth=settings.auth_enabled)
    logger.info(
        "Swedish Sentiment API starting up (auth=%s, ws_tickets=%s, transcription_events=%s)",
        settings.auth_enabled,
        app.state.ws_tickets.backend,
        hub.backend,
    )
    yield
    await hub.stop_redis_listener()
    reporter.phase("api", "shutdown", "Swedish Sentiment API shutting down")
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
        version=get_package_version(),
        description=(
            "REST API for Swedish sentiment analysis, ASR transcription, "
            "call-center conversation analysis, and batch processing."
        ),
        lifespan=lifespan,
    )

    # Starlette applies middleware LIFO. Add RateLimit first so RequestId is
    # outer and 429 responses still get X-Request-ID.
    if settings.rate_limit_rpm > 0:
        app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.rate_limit_rpm)
    app.add_middleware(RequestIdMiddleware)
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
        elif exc.status_code == 422:
            code = ERROR_CODE_VALIDATION
        elif exc.status_code == 413:
            code = ERROR_CODE_PAYLOAD_TOO_LARGE
        elif exc.status_code == 409:
            code = ERROR_CODE_CONFLICT
        elif exc.status_code == 403:
            code = ERROR_CODE_FORBIDDEN
        elif exc.status_code == 404:
            code = ERROR_CODE_NOT_FOUND
        else:
            code = ERROR_CODE_INTERNAL
        return error_response(request, exc.status_code, exc.detail, error_code=code)

    @app.exception_handler(LLMError)
    async def handle_llm_error(request: Request, exc: LLMError) -> JSONResponse:
        logger.error("LLM error: %s", exc)
        return error_response(
            request,
            502,
            public_detail(exc, dev_prefix="LLM request failed", public=LLM_ERROR_DETAIL),
            error_code=error_code_for(exc),
        )

    @app.exception_handler(ConfigurationError)
    async def handle_config_error(request: Request, exc: ConfigurationError) -> JSONResponse:
        logger.warning("Configuration error: %s", exc)
        return error_response(
            request,
            422,
            public_detail(exc, dev_prefix="Configuration error", public=CONFIGURATION_ERROR_DETAIL),
            error_code=error_code_for(exc),
        )

    @app.exception_handler(TranscriptionError)
    async def handle_transcription_error(request: Request, exc: TranscriptionError) -> JSONResponse:
        logger.error("Transcription error: %s", exc)
        return error_response(
            request,
            500,
            public_detail(
                exc, dev_prefix="Transcription failed", public=TRANSCRIPTION_ERROR_DETAIL
            ),
            error_code=error_code_for(exc),
        )

    @app.exception_handler(AnalysisError)
    async def handle_analysis_error(request: Request, exc: AnalysisError) -> JSONResponse:
        logger.error("Analysis error: %s", exc)
        return error_response(
            request,
            500,
            public_detail(exc, dev_prefix="Analysis failed", public=ANALYSIS_ERROR_DETAIL),
            error_code=error_code_for(exc),
        )

    @app.exception_handler(BaseAnalysisError)
    async def handle_base_error(request: Request, exc: BaseAnalysisError) -> JSONResponse:
        logger.error("Analysis system error: %s", exc)
        return error_response(
            request,
            500,
            public_detail(exc, public=PUBLIC_ERROR_DETAIL),
            error_code=error_code_for(exc),
            details=getattr(exc, "details", None) or None,
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled exception: %s", exc, exc_info=True)
        return error_response(
            request,
            500,
            PUBLIC_ERROR_DETAIL,
            error_code=ERROR_CODE_INTERNAL,
            details={"type": type(exc).__name__, "message": str(exc)},
        )

    # --- Routers -------------------------------------------------------------

    _auth = [Depends(require_api_key)]

    app.include_router(health.router)
    app.include_router(status.router, dependencies=_auth)
    app.include_router(text.router, dependencies=_auth)
    app.include_router(transcription.router, dependencies=_auth)
    app.include_router(conversation.router, dependencies=_auth)
    app.include_router(pipeline.router, dependencies=_auth)
    app.include_router(calls.router, dependencies=_auth)
    app.include_router(llm_profiles.router, dependencies=_auth)
    app.include_router(scan.router, dependencies=_auth)
    app.include_router(ws_transcription.router)
    app.include_router(alerting.router, dependencies=_auth)
    app.include_router(edge.router, dependencies=_auth)

    _init_app_state(app)
    return app


# Default application instance (used by uvicorn src.api:app etc.)
app: FastAPI = create_app()
