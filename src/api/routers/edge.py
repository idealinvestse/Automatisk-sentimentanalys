"""Edge AI offline analysis router.

Exposes the lightweight offline inference from `src/edge/local_inference.py`
over the REST API so the webui can run edge-mode analysis without the full
pipeline (no LLM, no registry analyzers, no Fas 4 enrichment).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ...edge.contracts import EdgeAnalysisResult
from ...edge.local_inference import analyze_segments_offline, analyze_text_offline
from ..router_errors import run_route

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/edge", tags=["Edge"])


# --- Request schemas --------------------------------------------------------


class EdgeTextRequest(BaseModel):
    """Single-text offline analysis request."""

    text: str = Field(..., min_length=1, description="Text to analyze offline")
    profile: str = Field("callcenter", description="Sentiment profile")


class EdgeSegmentsRequest(BaseModel):
    """Multi-segment offline analysis request (pre-transcribed)."""

    segments: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="Pre-transcribed segments with 'text' and optionally 'speaker'",
    )
    profile: str = Field("callcenter", description="Sentiment profile")


# --- Endpoints --------------------------------------------------------------


@router.post("/analyze-text", response_model=EdgeAnalysisResult)
async def analyze_text_edge(req: EdgeTextRequest) -> EdgeAnalysisResult:
    """Run offline sentiment + heuristic intent on a single text string.

    Equivalent to `sentimentanalys edge-analyze --text ...`.
    No LLM, no diarization, no Fas 4 — lightweight and offline-first.
    """
    logger.info("Edge text analysis: profile=%s len=%d", req.profile, len(req.text))

    async def _do() -> EdgeAnalysisResult:
        return await asyncio.to_thread(analyze_text_offline, req.text, profile=req.profile)

    return await run_route("edge/analyze-text", _do)


@router.post("/analyze-segments", response_model=EdgeAnalysisResult)
async def analyze_segments_edge(req: EdgeSegmentsRequest) -> EdgeAnalysisResult:
    """Run offline analysis on pre-transcribed segments.

    Equivalent to `sentimentanalys edge-analyze` on segment input.
    Applies early PII redaction (callcenter profile) + sentiment + heuristic intent.
    """
    logger.info("Edge segments analysis: profile=%s segments=%d", req.profile, len(req.segments))

    async def _do() -> EdgeAnalysisResult:
        return await asyncio.to_thread(analyze_segments_offline, req.segments, profile=req.profile)

    return await run_route("edge/analyze-segments", _do)
