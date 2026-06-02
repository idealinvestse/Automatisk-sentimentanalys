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
