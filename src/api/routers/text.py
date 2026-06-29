"""Text sentiment analysis router (/analyze)."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter

from ...core.serialization import utc_now_iso
from ...sentiment import analyze_smart
from ..router_errors import run_route
from ..schemas import AnalyzeRequest, AnalyzeResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Sentiment"])


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze sentiment of Swedish texts."""
    logger.info("Analyzing sentiment for %d text(s)", len(req.texts))

    async def _do() -> AnalyzeResponse:
        results, meta = await asyncio.to_thread(
            analyze_smart,
            texts=req.texts,
            datatype=req.datatype,
            source=req.source,
            profile=req.profile,
            model_name=req.model,
            device=req.device,
            batch_size=req.batch_size,
            normalize=req.normalize,
            return_all_scores=req.return_all_scores,
            max_length=req.max_length,
            clean=req.clean,
            lexicon_file=req.lexicon_file,
            lexicon_weight=req.lexicon_weight,
        )
        logger.info(
            "Analysis complete – profile=%s model=%s", meta.get("profile"), meta.get("model")
        )
        return AnalyzeResponse(meta=meta, timestamp=utc_now_iso(), results=results)

    return await run_route("analyze", _do)
