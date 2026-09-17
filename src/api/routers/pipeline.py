"""Full call-analysis pipeline router (/analyze_pipeline)."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request

from ...alerting import AlertEngine
from ...caching import AggregateCache
from ...core.serialization import utc_now_iso
from ...customers import CustomerContext
from ...pipeline import CallAnalysisPipeline
from ...profiles import resolve_profile
from ..call_persistence import customer_ref, get_call_store, persist_call_artifact, report_as_dict
from ..call_store import call_idempotency_key
from ..dependencies import (
    create_pipeline,
    get_alert_engine,
    get_cache,
    get_openrouter_header_key,
    resolve_llm_api_key,
)
from ..helpers import apply_customer_policy, apply_llm_ceiling, resolve_customer_or_422
from ..router_errors import run_route
from ..schemas import (
    AgentPerformanceRequest,
    AgentPerformanceResponse,
    AlertsRequest,
    AlertsResponse,
    AnalysisJobCancelResponse,
    AnalysisJobRequest,
    AnalysisJobStatusResponse,
    HotTopicsRequest,
    HotTopicsResponse,
    ModelCompareResult,
    PartialPipelineRequest,
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
    customer = _resolve_pipeline_customer(req)
    return _pipeline_from_request(req, cache, header_key, customer)


def _resolve_pipeline_customer(req: Any) -> CustomerContext | None:
    return resolve_customer_or_422(getattr(req, "original_filename", None))


def _pipeline_from_request(
    req: Any,
    cache: AggregateCache,
    header_key: str | None,
    customer: CustomerContext | None,
) -> CallAnalysisPipeline:
    llm_requested = bool(
        getattr(req, "use_mistral_llm", False) or getattr(req, "deep_analysis", False)
    )
    policy = apply_customer_policy(
        customer,
        requested_llm_enabled=llm_requested,
        requested_llm_provider=getattr(req, "provider", None),
        requested_profile=getattr(req, "profile", None),
    )
    pipe = create_pipeline(
        cache=cache,
        profile=policy.analyzer_profile,
        sentiment_model=getattr(req, "sentiment_model", None),
        device=getattr(req, "device", "auto"),
        use_mistral_llm=policy.llm_enabled and bool(getattr(req, "use_mistral_llm", False)),
        llm_model=getattr(req, "llm_model", None),
        deep_analysis=policy.llm_enabled and bool(getattr(req, "deep_analysis", False)),
        llm_api_key=resolve_llm_api_key(getattr(req, "llm_api_key", None), header_key),
        provider=policy.llm_provider or getattr(req, "provider", "openrouter"),
        groq_eu_residency=getattr(req, "groq_eu_residency", False),
        async_analyzers=getattr(req, "async_analyzers", False),
        analysis_perspective=getattr(req, "analysis_perspective", None)
        if policy.llm_enabled
        else None,
        qa_scorecard=policy.qa_scorecard,
        customer_id=policy.customer_id,
        config_fingerprint=policy.config_fingerprint,
    )
    apply_llm_ceiling(pipe, policy)
    return pipe


@router.post("/analyze_pipeline", response_model=PipelineResponse)
async def analyze_pipeline(
    req: PipelineRequest,
    request: Request,
    cache: Annotated[AggregateCache, Depends(get_cache)],
    header_key: Annotated[str | None, Depends(get_openrouter_header_key)] = None,
) -> PipelineResponse:
    """Run the full call analysis pipeline on pre-transcribed segments."""
    logger.info("Running full pipeline on %d segment(s)", len(req.segments))
    customer = _resolve_pipeline_customer(req)
    pipe = _pipeline_from_request(req, cache, header_key, customer)

    async def _do() -> PipelineResponse:
        report = await asyncio.to_thread(
            pipe.analyze_segments,
            req.segments,
            req.selected_analyzers,
        )
        stored = persist_call_artifact(
            get_call_store(request),
            status="completed",
            call_id=req.call_id,
            transcript={"segments": req.segments},
            report=report_as_dict(report),
            customer=customer,
            original_filename=req.original_filename,
            route="analyze_pipeline",
        )
        return _report_to_pipeline_response(
            report,
            customer=customer,
            call_id=stored.get("id"),
            persisted=True,
        )

    return await run_route("analyze_pipeline", _do)


@router.post("/analysis/jobs", response_model=AnalysisJobStatusResponse, status_code=202)
async def create_analysis_job(
    req: AnalysisJobRequest,
    request: Request,
    cache: Annotated[AggregateCache, Depends(get_cache)],
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> AnalysisJobStatusResponse:
    """Submit a PII-redacted long-context LM Studio analysis."""
    if idempotency_key and len(idempotency_key) > 128:
        raise HTTPException(
            status_code=400, detail="Idempotency-Key must be at most 128 characters"
        )
    customer = _resolve_pipeline_customer(req)
    policy = apply_customer_policy(
        customer,
        requested_llm_enabled=True,
        requested_llm_provider="lmstudio",
        requested_profile=req.profile,
    )
    try:
        from ...llm.pii_redactor import redact_segments

        redacted_segments, pii_log = redact_segments(
            req.segments,
            profile_name=policy.analyzer_profile,
            return_log=True,
            force=True,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail="PII redaction failed; analysis job rejected"
        ) from exc
    if pii_log.error:
        raise HTTPException(status_code=422, detail="PII redaction failed; analysis job rejected")

    payload = req.model_dump()
    payload["segments"] = redacted_segments
    payload["profile"] = policy.analyzer_profile
    store = get_call_store(request)
    persist_call_artifact(
        store,
        status="transcribed",
        call_id=req.call_id,
        transcript={"segments": redacted_segments},
        customer=customer,
        original_filename=req.original_filename,
        route="analysis_jobs",
    )

    def _runner(job_payload: dict[str, Any]) -> dict[str, Any]:
        pipe = create_pipeline(
            cache=cache,
            profile=policy.analyzer_profile,
            sentiment_model=job_payload.get("sentiment_model"),
            device="cpu",
            use_mistral_llm=True,
            llm_model=job_payload.get("llm_model"),
            deep_analysis=True,
            provider="lmstudio",
            qa_scorecard=policy.qa_scorecard,
            customer_id=policy.customer_id,
            config_fingerprint=policy.config_fingerprint,
        )
        report = pipe.analyze_segments(
            job_payload["segments"],
            job_payload.get("selected_analyzers"),
        )
        stored = persist_call_artifact(
            store,
            status="completed",
            call_id=req.call_id,
            transcript={"segments": job_payload["segments"]},
            report=report_as_dict(report),
            customer=customer,
            original_filename=req.original_filename,
            route="analysis_jobs",
        )
        return _report_to_pipeline_response(
            report,
            customer=customer,
            call_id=stored.get("id"),
            persisted=True,
        ).model_dump(mode="json")

    job_key = idempotency_key or call_idempotency_key(
        policy.customer_id,
        req.original_filename or "",
        policy.config_fingerprint,
    )
    try:
        job = request.app.state.analysis_jobs.submit(
            payload,
            _runner,
            idempotency_key=job_key,
        )
    except OverflowError as exc:
        raise HTTPException(status_code=429, detail="Analysis job queue is full") from exc
    return AnalysisJobStatusResponse(**job.to_dict())


@router.get("/analysis/jobs/{job_id}", response_model=AnalysisJobStatusResponse)
async def get_analysis_job(job_id: str, request: Request) -> AnalysisJobStatusResponse:
    """Return status for a long-context analysis job."""
    job = request.app.state.analysis_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Analysis job not found")
    return AnalysisJobStatusResponse(**job.to_dict())


@router.get("/analysis/jobs/{job_id}/result", response_model=PipelineResponse)
async def get_analysis_job_result(job_id: str, request: Request) -> PipelineResponse:
    """Return the persisted report for a completed analysis job."""
    job = request.app.state.analysis_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Analysis job not found")
    result = request.app.state.analysis_jobs.result(job_id)
    if result is None:
        raise HTTPException(status_code=409, detail=f"Analysis result not ready: {job.status}")
    return PipelineResponse.model_validate(result)


@router.post("/analysis/jobs/{job_id}/cancel", response_model=AnalysisJobCancelResponse)
async def cancel_analysis_job(job_id: str, request: Request) -> AnalysisJobCancelResponse:
    """Request cancellation without releasing an active inference slot early."""
    outcome = request.app.state.analysis_jobs.cancel(job_id)
    if outcome == "not_found":
        raise HTTPException(status_code=404, detail="Analysis job not found")
    if outcome == "already_finished":
        raise HTTPException(status_code=409, detail="Analysis job already finished")
    return AnalysisJobCancelResponse(job_id=job_id, status=outcome)


@router.post("/analyze_pipeline/partial", response_model=PipelineResponse)
async def analyze_pipeline_partial(
    req: PartialPipelineRequest,
    cache: Annotated[AggregateCache, Depends(get_cache)],
    header_key: Annotated[str | None, Depends(get_openrouter_header_key)] = None,
) -> PipelineResponse:
    """Incremental local analysis with optional holistic LLM reconciliation."""
    logger.info(
        "Running partial pipeline on %d segment(s) reconcile=%s",
        len(req.segments),
        req.reconcile,
    )
    customer = _resolve_pipeline_customer(req)
    pipe = _pipeline_from_request(req, cache, header_key, customer)

    async def _do() -> PipelineResponse:
        report = await asyncio.to_thread(
            pipe.analyze_segments_partial,
            req.segments,
            previous_results=req.previous_results,
            selected_analyzers=req.selected_analyzers,
            reconcile=req.reconcile,
        )
        return _report_to_pipeline_response(report, customer=customer, call_id=req.call_id)

    return await run_route("analyze_pipeline_partial", _do)


def _report_to_pipeline_response(
    report: Any,
    *,
    customer: CustomerContext | None = None,
    call_id: str | None = None,
    persisted: bool = False,
) -> PipelineResponse:
    from ..degradation import collect_degraded_reasons

    degraded = collect_degraded_reasons(report)
    return PipelineResponse(
        sentiment_results=report.sentiment_results,
        intent_results=[{"intent": i, "confidence": round(c, 3)} for i, c in report.intent_results],
        summary=report.summary,
        topics=report.topics,
        insights=report.insights,
        risks=report.risks,
        processing_time_s=report.processing_time_s,
        timestamp=utc_now_iso(),
        segments=list(report.segments) if isinstance(report.segments, list) else [],
        diarization=report.diarization if isinstance(report.diarization, dict) else None,
        llm=report.llm,
        results=report.results,
        analyzer_results=build_analyzer_results(report.results),
        degraded=degraded,
        mode="degraded" if degraded else "full",
        customer=customer_ref(customer),
        call_id=call_id,
        persisted=persisted,
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
    customer = _resolve_pipeline_customer(req)

    async def _do() -> PipelineCompareResponse:
        nonlocal total_cost, total_time, budget_exceeded
        for model in req.models:
            if budget_exceeded:
                break
            pipe = _pipeline_from_request(
                SimpleNamespace(
                    profile=req.profile,
                    sentiment_model=req.sentiment_model,
                    device=req.device,
                    use_mistral_llm=True,
                    llm_model=model,
                    deep_analysis=req.deep_analysis,
                    llm_api_key=req.llm_api_key,
                    provider=req.provider,
                    groq_eu_residency=req.groq_eu_residency,
                    async_analyzers=False,
                    analysis_perspective=None,
                    original_filename=req.original_filename,
                ),
                cache,
                header_key,
                customer,
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
            qa = (report.results or {}).get("qa") or (report.results or {}).get("compliance_qa", {})
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
