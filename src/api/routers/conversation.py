"""Conversation analysis routers (/analyze_conversation, /batch_analyze_conversation)."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from ...caching import AggregateCache
from ...core.serialization import utc_now_iso
from ..batch import run_batch
from ..call_persistence import get_call_store
from ..dependencies import get_cache
from ..helpers import resolve_customer_context, resolve_customer_or_422
from ..path_validation import resolve_and_validate_audio_paths
from ..router_errors import run_route
from ..schemas import (
    AnalyzeConversationRequest,
    AnalyzeConversationResponse,
    BatchAnalyzeConversationItem,
    BatchAnalyzeConversationRequest,
    BatchAnalyzeConversationResponse,
)
from ..services.conversation import run_analyze_conversation, run_batch_analyze_file

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Conversation"])


@router.post("/analyze_conversation", response_model=AnalyzeConversationResponse)
async def analyze_conversation(
    req: AnalyzeConversationRequest,
    request: Request,
    cache: Annotated[AggregateCache, Depends(get_cache)],
) -> AnalyzeConversationResponse:
    """Transcribe a call and run sentiment analysis per segment."""
    logger.info(
        "Analyzing conversation: %s (backend=%s model=%s full_pipeline=%s)",
        req.audio_path,
        req.backend,
        req.model,
        req.use_full_pipeline,
    )

    # Original filename is authoritative for customer identity; the stored
    # (uuid-prefixed, sanitized) name is only a fallback.
    customer = resolve_customer_or_422(req.original_filename or Path(req.audio_path).name)
    store = get_call_store(request)

    async def _do() -> AnalyzeConversationResponse:
        resp = await asyncio.to_thread(
            run_analyze_conversation,
            req,
            cache=cache,
            customer=customer,
            call_store=store,
        )
        if customer is not None:
            resp.meta["customer"] = customer.model_dump(mode="json")
        return resp

    return await run_route("analyze_conversation", _do)


@router.post("/batch_analyze_conversation", response_model=BatchAnalyzeConversationResponse)
async def batch_analyze_conversation(
    req: BatchAnalyzeConversationRequest,
) -> BatchAnalyzeConversationResponse:
    """Analyze sentiment for multiple conversation audio files."""

    async def _do() -> BatchAnalyzeConversationResponse:
        files = resolve_and_validate_audio_paths(
            audio_paths=req.audio_paths,
            directory=req.directory,
            pattern=req.glob,
            recursive=req.recursive,
            limit=req.limit,
        )
        logger.info(
            "Batch analyzing %d conversation file(s) with %d worker(s)", len(files), req.workers
        )

        def _worker(p: str) -> tuple:
            # Per-file customer gate: controlled modes reject unknown/ambiguous
            # identities per item instead of aborting the whole batch.
            ctx = resolve_customer_context(Path(p).name)
            tr, segs, meta, _pipe = run_batch_analyze_file(req, p, customer=ctx)
            if ctx is not None:
                meta = dict(meta or {})
                meta["customer"] = ctx.model_dump(mode="json")
            return tr, segs, meta

        raw = await asyncio.to_thread(
            run_batch,
            files,
            _worker,
            workers=req.workers,
            worker_timeout=req.worker_timeout,
        )
        items: list[BatchAnalyzeConversationItem] = []
        ok = failed = 0
        for path, result, error in raw:
            if error is None and result is not None:
                tr, segs, meta = result
                items.append(
                    BatchAnalyzeConversationItem(
                        file=path,
                        transcript=tr,
                        segment_sentiments=segs,
                        meta=meta,
                    )
                )
                ok += 1
            else:
                items.append(BatchAnalyzeConversationItem(file=path, error=str(error)))
                failed += 1
        return BatchAnalyzeConversationResponse(
            items=items, ok=ok, failed=failed, total=len(files), timestamp=utc_now_iso()
        )

    return await run_route("batch_analyze_conversation", _do)
