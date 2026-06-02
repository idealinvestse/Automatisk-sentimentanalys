"""Text sentiment analysis router (/analyze)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from ...core.serialization import utc_now_iso
from ...sentiment import analyze_smart
from ..schemas import AnalyzeRequest, AnalyzeResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Sentiment"])


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze sentiment of Swedish texts.

    Supports profile-based analysis (forum, news, callcenter, etc.) and
    optional lexicon blending for improved accuracy with domain-specific
    language.

    Returns:
        Sentiment results, meta-information, and a UTC timestamp.
    """
    logger.info("Analyzing sentiment for %d text(s)", len(req.texts))
    try:
        results, meta = analyze_smart(
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
    except Exception as e:
        logger.error("Sentiment analysis failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {e}") from e

    logger.info("Analysis complete – profile=%s model=%s", meta.get("profile"), meta.get("model"))
    return AnalyzeResponse(meta=meta, timestamp=utc_now_iso(), results=results)
