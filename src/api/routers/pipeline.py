"""Full call-analysis pipeline router (/analyze_pipeline)."""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Any

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
    ModelCompareResult,
    PipelineCompareRequest,
    PipelineCompareResponse,
    PipelineRequest,
    PipelineResponse,
    QAScoreRequest,
    QAScoreResponse,
    SemanticSearchRequest,
    SemanticSearchResponse,
    build_analyzer_results,
)
from ...profiles import resolve_profile
from ..services.pipeline_cache import resolve_reports

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Pipeline"])


def _fas4_pipeline(
    req: (
        AgentPerformanceRequest
        | SemanticSearchRequest
        | HotTopicsRequest
        | QAScoreRequest
        | AlertsRequest
    ),
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
        provider=getattr(req, "provider", "openrouter"),
        groq_eu_residency=getattr(req, "groq_eu_residency", False),
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
        profile=req.profile,
        sentiment_model=req.sentiment_model,
        device=req.device,
        use_mistral_llm=req.use_mistral_llm,
        llm_model=req.llm_model,
        deep_analysis=req.deep_analysis,
        llm_api_key=resolve_llm_api_key(req.llm_api_key, header_key),
        provider=req.provider,
        groq_eu_residency=req.groq_eu_residency,
        async_analyzers=req.async_analyzers,
    )

    async def _do() -> PipelineResponse:
        report = await asyncio.to_thread(
            pipe.analyze_segments,
            req.segments,
            req.selected_analyzers,
        )
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
            analyzer_results=build_analyzer_results(report.results),
        )

    return await run_route("analyze_pipeline", _do)


def _report_to_pipeline_response(report: Any) -> PipelineResponse:
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
        analyzer_results=build_analyzer_results(report.results),
    )


def _extract_llm_cost(report: Any) -> float | None:
    llm = report.llm if isinstance(report.llm, dict) else {}
    meta = llm.get("meta") if isinstance(llm, dict) else None
    if isinstance(meta, dict):
        cost = meta.get("cost_usd") or meta.get("cost")
        if cost is not None:
            try:
                return float(cost)
            except (TypeError, ValueError):
                return None
    results = report.results if isinstance(report.results, dict) else {}
    judge = results.get("llm_judge")
    if isinstance(judge, dict):
        cost = judge.get("total_cost_usd") or judge.get("cost_usd")
        if cost is not None:
            try:
                return float(cost)
            except (TypeError, ValueError):
                return None
    return None


def _resolve_compare_budget(req: PipelineCompareRequest) -> float:
    if req.cost_budget_usd is not None:
        return req.cost_budget_usd
    _, spec = resolve_profile(profile=req.profile)
    llm_cfg = spec.get("llm") or {}
    return float(llm_cfg.get("cost_budget_per_call", 0.08))


@router.post("/analyze_pipeline/compare", response_model=PipelineCompareResponse)
async def analyze_pipeline_compare(
    req: PipelineCompareRequest,
    cache: Annotated[AggregateCache, Depends(get_cache)],
    header_key: Annotated[str | None, Depends(get_openrouter_header_key)] = None,
) -> PipelineCompareResponse:
    """Run the same segments through up to 3 LLM models for side-by-side comparison."""
    logger.info(
        "Pipeline compare on %d segment(s), models=%s",
        len(req.segments),
        req.models,
    )
    budget = _resolve_compare_budget(req)
    per_model_budget = budget / max(len(req.models), 1)
    results: dict[str, ModelCompareResult] = {}
    total_cost = 0.0
    total_time = 0.0
    budget_exceeded = False

    async def _do() -> PipelineCompareResponse:
        nonlocal total_cost, total_time, budget_exceeded
        for model in req.models:
            if budget_exceeded:
                break
            pipe = create_pipeline(
                cache=cache,
                profile=req.profile,
                sentiment_model=req.sentiment_model,
                device=req.device,
                use_mistral_llm=True,
                llm_model=model,
                deep_analysis=req.deep_analysis,
                llm_api_key=resolve_llm_api_key(req.llm_api_key, header_key),
                provider=req.provider,
                groq_eu_residency=req.groq_eu_residency,
            )
            report = await asyncio.to_thread(
                pipe.analyze_segments,
                req.segments,
                req.selected_analyzers,
            )
            response = _report_to_pipeline_response(report)
            cost = _extract_llm_cost(report) or 0.0
            total_cost += cost
            total_time += report.processing_time_s
            if total_cost > budget:
                budget_exceeded = True
            qa = (report.results or {}).get("qa") or (report.results or {}).get(
                "compliance_qa", {}
            )
            qa_score = qa.get("overall_qa_score") if isinstance(qa, dict) else None
            sentiment_label = None
            if report.sentiment_results:
                first = report.sentiment_results[0]
                if isinstance(first, dict):
                    sentiment_label = first.get("label")
            llm_traj = None
            if isinstance(report.llm, dict):
                traj = report.llm.get("trajectory")
                if isinstance(traj, dict):
                    llm_traj = traj.get("trend") or traj.get("summary")
            results[model] = ModelCompareResult(
                model=model,
                processing_time_s=report.processing_time_s,
                llm_cost_usd=cost if cost else None,
                qa_score=float(qa_score) if qa_score is not None else None,
                sentiment_label=sentiment_label,
                llm_trajectory=str(llm_traj) if llm_traj is not None else None,
                response=response,
            )
            if cost > per_model_budget:
                logger.warning(
                    "Model %s cost %.4f exceeds per-model budget %.4f",
                    model,
                    cost,
                    per_model_budget,
                )
        return PipelineCompareResponse(
            models=req.models,
            results=results,
            total_cost_usd=round(total_cost, 6) if total_cost else None,
            total_processing_time_s=round(total_time, 3),
            budget_usd=budget,
            budget_exceeded=budget_exceeded,
            timestamp=utc_now_iso(),
        )

    return await run_route("analyze_pipeline_compare", _do)


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
