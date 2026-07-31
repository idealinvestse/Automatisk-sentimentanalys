"""FastAPI dependencies: auth, shared cache, pipeline factory (Fas 2)."""

from __future__ import annotations

import logging
from typing import Annotated, cast

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from ..alerting import AlertEngine
from ..caching import AggregateCache
from ..pipeline import CallAnalysisPipeline
from .settings import get_api_settings

logger = logging.getLogger(__name__)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_openrouter_header = APIKeyHeader(name="X-OpenRouter-Key", auto_error=False)


async def require_api_key(
    api_key: Annotated[str | None, Security(_api_key_header)] = None,
) -> None:
    """Optional API key when SENTIMENT_API_KEY is unset (local dev / tests)."""
    settings = get_api_settings()
    if not settings.auth_enabled:
        return
    if not api_key or api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def get_cache(request: Request) -> AggregateCache:
    return cast(AggregateCache, request.app.state.cache)


def get_alert_engine(request: Request) -> AlertEngine:
    return cast(AlertEngine, request.app.state.alert_engine)


def get_openrouter_header_key(
    header_key: Annotated[str | None, Security(_openrouter_header)] = None,
) -> str | None:
    return header_key


def resolve_llm_api_key(
    body_key: str | None,
    header_key: str | None = None,
) -> str | None:
    """Prefer header over body; body only when explicitly allowed."""
    if header_key:
        return header_key
    settings = get_api_settings()
    if body_key and settings.allow_client_llm_key_in_body:
        return body_key
    if body_key and not settings.allow_client_llm_key_in_body:
        logger.warning(
            "Ignoring llm_api_key in request body (set API_ALLOW_CLIENT_LLM_KEY=true to allow)"
        )
    return None


def create_pipeline(
    *,
    cache: AggregateCache,
    profile: str = "default",
    sentiment_model: str | None = None,
    device: str = "auto",
    use_mistral_llm: bool = False,
    llm_model: str | None = None,
    deep_analysis: bool = False,
    llm_api_key: str | None = None,
    provider: str = "openrouter",
    groq_eu_residency: bool = False,
    async_analyzers: bool = False,
    analysis_perspective: str | None = None,
) -> CallAnalysisPipeline:
    # Cost/quality-aware paid model pick from analysis perspective menu
    if analysis_perspective and not llm_model:
        try:
            from ..llm.paid_model_advisor import recommend_for_perspective

            rec = recommend_for_perspective(analysis_perspective, top_k=1)
            sel = rec.selectable or {}
            if sel.get("llm_model"):
                llm_model = str(sel["llm_model"])
            if sel.get("provider") and provider in {
                "openrouter",
                "auto",
                "router",
                "sv_optimal",
                "free_sequential",
            }:
                # Only override generic providers; honor explicit mistral/nvidia/…
                provider = str(sel["provider"])
            if sel.get("deep_analysis"):
                deep_analysis = True
            use_mistral_llm = True
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning(
                "analysis_perspective=%s resolve failed: %s", analysis_perspective, exc
            )

    return CallAnalysisPipeline(
        sentiment_model=sentiment_model or "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        device=device,
        profile=profile,
        use_mistral_llm=use_mistral_llm,
        llm_model=llm_model,
        deep_analysis=deep_analysis,
        llm_api_key=llm_api_key,
        provider=provider,
        groq_eu_residency=groq_eu_residency,
        cache=cache,
        async_analyzers=async_analyzers,
    )
