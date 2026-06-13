"""Full call-analysis pipeline router (/analyze_pipeline)."""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from ...alerting import AlertEngine
from ...caching import AggregateCache
from ...core.serialization import utc_now_iso
from ...pipeline import CallAnalysisPipeline
from ..dependencies import (
    create_pipeline,
    get_alert_engine,
    get_cache,
    get_openrouter_header_key,
    resolve_llm_api_key,
)
from ..router_errors import run_route
from ..schemas import (
    AgentPerformanceRequest,
    AgentPerformanceResponse,
    AlertsRequest,
    AlertsResponse,
    HotTopicsRequest,
    HotTopicsResponse,
    PipelineRequest,
    PipelineResponse,
    QAScoreRequest,
    QAScoreResponse,
    SemanticSearchRequest,
    SemanticSearchResponse,
)
from ..services.pipeline_cache import resolve_reports

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Pipeline"])


def _fas4_pipeline(
    req: AgentPerformanceRequest
    | SemanticSearchRequest
    | HotTopicsRequest
    | QAScoreRequest
    | AlertsRequest,
    cache: AggregateCache,
    header_key: str | None,
) -> CallAnalysisPipeline:
    return create_pipeline(
        cache=cache,
        profile=req.profile,
        use_mistral_llm=req.use_mistral_llm,
        llm_model=req.llm_model,
        deep_analysis=req.deep_analysis,
        llm_api_key=resolve_llm_api_key(req.llm_api_key, header_key),
    )


@router.post("/analyze_pipeline", response_model=PipelineResponse)
async def analyze_pipeline(
    req: PipelineRequest,
    cache: Annotated[AggregateCache, Depends(get_cache)],
    header_key: Annotated[str | None, Depends(get_openrouter_header_key)] = None,
) -> PipelineResponse:
    """Run the full call analysis pipeline on pre-transcribed segments."""
    logger.info("Running full pipeline on %d segment(s)", len(req.segments))
    pipe = create_pipeline(
        cache=cache,
        sentiment_model=req.sentiment_model,
        device=req.device,
        use_mistral_llm=req.use_mistral_llm,
        llm_model=req.llm_model,
        deep_analysis=req.deep_analysis,
        llm_api_key=resolve_llm_api_key(req.llm_api_key, header_key),
    )

    async def _do() -> PipelineResponse:
        report = await asyncio.to_thread(pipe.analyze_segments, req.segments)
        return PipelineResponse(
            sentiment_results=report.sentiment_results,
            intent_results=[
                {"intent": i, "confidence": round(c, 3)} for i, c in report.intent_results
            ],
            summary=report.summary,
            topics=report.topics,
            insights=report.insights,
            risks=report.risks,
            processing_time_s=report.processing_time_s,
            timestamp=utc_now_iso(),
            llm=report.llm,
            results=report.results,
        )

    return await run_route("analyze_pipeline", _do)


@router.post("/agent_performance/{agent_id}", response_model=AgentPerformanceResponse)
async def get_agent_performance(
    agent_id: str,
    req: AgentPerformanceRequest,
    cache: Annotated[AggregateCache, Depends(get_cache)],
    header_key: Annotated[str | None, Depends(get_openrouter_header_key)] = None,
) -> AgentPerformanceResponse:
    """Get pre-computed/cached agent performance aggregates (Fas 4.5.1 + 4.5.2)."""
    if req.agent_id != agent_id:
        raise HTTPException(
            status_code=422,
            detail="Path agent_id must match body agent_id",
        )
    logger.info("Agent performance request for %s, %d calls", agent_id, len(req.segments_list))
    pipe = _fas4_pipeline(req, cache, header_key)

    async def _do() -> AgentPerformanceResponse:
        reports, _ = await asyncio.to_thread(
            resolve_reports, pipe, req.segments_list, reanalyze=req.reanalyze
        )
        metrics = dict(pipe.get_cached_agent_performance(agent_id, reports, window=req.window))
        cached = bool(metrics.pop("cache_hit", False))
        return AgentPerformanceResponse(
            agent_id=agent_id,
            metrics=metrics,
            cached=cached,
            timestamp=utc_now_iso(),
        )

    return await run_route("agent_performance", _do)


@router.post("/search/semantic", response_model=SemanticSearchResponse)
async def semantic_search(
    req: SemanticSearchRequest,
    cache: Annotated[AggregateCache, Depends(get_cache)],
    header_key: Annotated[str | None, Depends(get_openrouter_header_key)] = None,
) -> SemanticSearchResponse:
    """Hybrid semantic + keyword search over provided calls (Fas 4.3.2 + 4.5.2)."""
    logger.info("Semantic search: %s", req.query[:50])
    pipe = _fas4_pipeline(req, cache, header_key)

    async def _do() -> SemanticSearchResponse:
        reports, _ = await asyncio.to_thread(
            resolve_reports, pipe, req.segments_list, reanalyze=req.reanalyze
        )
        hits = pipe.semantic_search(
            req.query, top_k=req.top_k, filters=req.filters or {}, corpus=reports
        )
        return SemanticSearchResponse(
            query=req.query,
            hits=hits.get("hits", []),
            meta=hits.get("meta", {}),
            timestamp=utc_now_iso(),
        )

    return await run_route("semantic_search", _do)


@router.post("/insights/hot_topics", response_model=HotTopicsResponse)
async def get_hot_topics(
    req: HotTopicsRequest,
    cache: Annotated[AggregateCache, Depends(get_cache)],
    header_key: Annotated[str | None, Depends(get_openrouter_header_key)] = None,
) -> HotTopicsResponse:
    """Get cached hot topics and trends (Fas 4.3.1 + 4.5.2)."""
    logger.info("Hot topics request, window=%s, calls=%d", req.window, len(req.segments_list))
    pipe = _fas4_pipeline(req, cache, header_key)

    async def _do() -> HotTopicsResponse:
        reports, _ = await asyncio.to_thread(
            resolve_reports, pipe, req.segments_list, reanalyze=req.reanalyze
        )
        topics = dict(pipe.get_cached_hot_topics(reports, window=req.window))
        topics.pop("cache_hit", None)
        return HotTopicsResponse(
            hot_topics=topics.get("hot_topics", []),
            meta=topics.get("meta", {}),
            timestamp=utc_now_iso(),
        )

    return await run_route("hot_topics", _do)


@router.post("/qa/score", response_model=QAScoreResponse)
async def get_qa_score(
    req: QAScoreRequest,
    cache: Annotated[AggregateCache, Depends(get_cache)],
    header_key: Annotated[str | None, Depends(get_openrouter_header_key)] = None,
) -> QAScoreResponse:
    """Run QA scoring on segments (Fas 4.2 + 4.5.2)."""
    pipe = _fas4_pipeline(req, cache, header_key)

    async def _do() -> QAScoreResponse:
        reports, _ = await asyncio.to_thread(
            resolve_reports, pipe, [req.segments], reanalyze=req.reanalyze
        )
        report = reports[0]
        qa = report.results.get("qa") or report.results.get("compliance_qa", {})
        return QAScoreResponse(qa=qa, timestamp=utc_now_iso())

    return await run_route("qa_score", _do)


@router.post("/alerts", response_model=AlertsResponse)
async def get_alerts(
    req: AlertsRequest,
    cache: Annotated[AggregateCache, Depends(get_cache)],
    alert_engine: Annotated[AlertEngine, Depends(get_alert_engine)],
    header_key: Annotated[str | None, Depends(get_openrouter_header_key)] = None,
) -> AlertsResponse:
    """Get alerts from per-call results or aggregate trends (Fas 4.4.2 + 4.5.2)."""
    pipe = _fas4_pipeline(req, cache, header_key)

    async def _do() -> AlertsResponse:
        alerts: list[dict] = []
        if req.segments_list:
            reports, _ = await asyncio.to_thread(
                resolve_reports, pipe, req.segments_list, reanalyze=req.reanalyze
            )
            for r in reports:
                alerts.extend(r.results.get("alerts", []))
        if req.aggregate:
            trend_alerts = alert_engine.check_from_aggregate(req.aggregate)
            for a in trend_alerts:
                if hasattr(a, "model_dump"):
                    alerts.append(a.model_dump())
                elif isinstance(a, dict):
                    alerts.append(a)
                else:
                    alerts.append({"detail": str(a)})
        return AlertsResponse(alerts=alerts, timestamp=utc_now_iso())

    return await run_route("alerts", _do)
