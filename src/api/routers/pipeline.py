"""Full call-analysis pipeline router (/analyze_pipeline)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from ...core.serialization import utc_now_iso
from ...pipeline import CallAnalysisPipeline
from ..schemas import PipelineRequest, PipelineResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Pipeline"])


@router.post("/analyze_pipeline", response_model=PipelineResponse)
async def analyze_pipeline(req: PipelineRequest) -> PipelineResponse:
    """Run the full call analysis pipeline on pre-transcribed segments.

    Orchestrates sentiment, intent, summarization, topic modelling,
    insights, and predictive analytics in a single call.

    Returns:
        Complete analysis results with a UTC timestamp.
    """
    logger.info("Running full pipeline on %d segment(s)", len(req.segments))
    try:
        pipe = CallAnalysisPipeline(
            sentiment_model=req.sentiment_model or "cardiffnlp/twitter-xlm-roberta-base-sentiment",
            device=req.device,
            use_mistral_llm=req.use_mistral_llm,
            llm_model=req.llm_model,
            deep_analysis=req.deep_analysis,
        )
        report = pipe.analyze_segments(req.segments)
    except Exception as e:
        logger.error("Pipeline analysis failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline analysis failed: {e}") from e

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
        results=report.results,  # Fas 4: agent_performance, qa/compliance_qa, agent_assessment, customer_metrics etc.
    )


# ---------------------------------------------------------------------------
# Fas 4.5.2 new endpoints - explicit integration with pipeline caching/aggregates/search/alerts
# ---------------------------------------------------------------------------

from ..schemas import (
    AgentPerformanceRequest, AgentPerformanceResponse,
    SemanticSearchRequest, SemanticSearchResponse,
    HotTopicsRequest, HotTopicsResponse,
    QAScoreRequest, QAScoreResponse,
    AlertsRequest, AlertsResponse,
)


@router.post("/agent_performance/{agent_id}", response_model=AgentPerformanceResponse)
async def get_agent_performance(agent_id: str, req: AgentPerformanceRequest) -> AgentPerformanceResponse:
    """Get pre-computed/cached agent performance aggregates (Fas 4.5.1 + 4.5.2).

    Internally runs pipeline on provided segments_list, then uses cached aggregates.
    """
    logger.info("Agent performance request for %s, %d calls", agent_id, len(req.segments_list))
    try:
        pipe = CallAnalysisPipeline(
            profile=req.profile,
            use_mistral_llm=req.use_mistral_llm,
            deep_analysis=req.use_mistral_llm,
        )
        reports = [pipe.analyze_segments(segs) for segs in req.segments_list]
        metrics = pipe.get_cached_agent_performance(agent_id, reports, window=req.window)
        # Check if it came from cache (simplified)
        cached = "computed_at" in metrics and metrics.get("call_count", 0) > 0
        return AgentPerformanceResponse(
            agent_id=agent_id,
            metrics=metrics,
            cached=cached,
            timestamp=utc_now_iso(),
        )
    except Exception as e:
        logger.error("Agent performance endpoint failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/search/semantic", response_model=SemanticSearchResponse)
async def semantic_search(req: SemanticSearchRequest) -> SemanticSearchResponse:
    """Hybrid semantic + keyword search over provided calls (Fas 4.3.2 + 4.5.2)."""
    logger.info("Semantic search: %s", req.query[:50])
    try:
        pipe = CallAnalysisPipeline(profile=req.profile)
        reports = [pipe.analyze_segments(segs) for segs in req.segments_list]
        hits = pipe.semantic_search(req.query, top_k=req.top_k, filters=req.filters or {}, corpus=reports)
        return SemanticSearchResponse(
            query=req.query,
            hits=hits.get("hits", []),
            meta=hits.get("meta", {}),
            timestamp=utc_now_iso(),
        )
    except Exception as e:
        logger.error("Semantic search failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/insights/hot_topics", response_model=HotTopicsResponse)
async def get_hot_topics(req: HotTopicsRequest) -> HotTopicsResponse:
    """Get cached hot topics and trends (Fas 4.3.1 + 4.5.2). Uses pre-computation cache."""
    logger.info("Hot topics request, window=%s, calls=%d", req.window, len(req.segments_list))
    try:
        pipe = CallAnalysisPipeline(
            profile=req.profile,
            use_mistral_llm=req.use_mistral_llm,
        )
        reports = [pipe.analyze_segments(segs) for segs in req.segments_list]
        topics = pipe.get_cached_hot_topics(reports, window=req.window)
        return HotTopicsResponse(
            hot_topics=topics.get("hot_topics", []),
            meta=topics.get("meta", {}),
            timestamp=utc_now_iso(),
        )
    except Exception as e:
        logger.error("Hot topics failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/qa/score", response_model=QAScoreResponse)
async def get_qa_score(req: QAScoreRequest) -> QAScoreResponse:
    """Run QA scoring on segments (Fas 4.2 + 4.5.2)."""
    try:
        pipe = CallAnalysisPipeline(profile=req.profile, use_mistral_llm=req.use_mistral_llm)
        report = pipe.analyze_segments(req.segments)
        qa = report.results.get("qa") or report.results.get("compliance_qa", {})
        return QAScoreResponse(qa=qa, timestamp=utc_now_iso())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/alerts", response_model=AlertsResponse)
async def get_alerts(req: AlertsRequest) -> AlertsResponse:
    """Get alerts from per-call results or aggregate trends (Fas 4.4.2 + 4.5.2)."""
    try:
        pipe = CallAnalysisPipeline(profile=req.profile)
        alerts: list[dict] = []
        if req.segments_list:
            for segs in req.segments_list:
                r = pipe.analyze_segments(segs)
                alerts.extend(r.results.get("alerts", []))
        if req.aggregate:
            # Simulate from aggregator data
            from ...alerting import AlertEngine
            eng = AlertEngine()
            trend_alerts = eng.check_from_aggregate(req.aggregate)
            alerts.extend([a.model_dump() if hasattr(a, "model_dump") else a for a in trend_alerts])
        return AlertsResponse(alerts=alerts, timestamp=utc_now_iso())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
