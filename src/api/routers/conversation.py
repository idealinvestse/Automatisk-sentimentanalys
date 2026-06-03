"""Conversation analysis routers (/analyze_conversation, /batch_analyze_conversation)."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from ...core.audio import resolve_audio_paths
from ...core.serialization import map_results_to_segment_dicts, texts_from_segments, utc_now_iso
from ...sentiment import analyze_smart
from ..batch import run_batch
from ..helpers import transcribe_helper
from ..router_errors import run_route
from ..schemas import (
    AnalyzeConversationRequest,
    AnalyzeConversationResponse,
    BatchAnalyzeConversationItem,
    BatchAnalyzeConversationRequest,
    BatchAnalyzeConversationResponse,
    SegmentSentiment,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Conversation"])


def _build_segment_sentiments(
    texts: list[str],
    results: list,
    segments: list[dict],
) -> list[SegmentSentiment]:
    dicts = map_results_to_segment_dicts(texts, results, segments)
    return [SegmentSentiment(**d) for d in dicts]


@router.post("/analyze_conversation", response_model=AnalyzeConversationResponse)
async def analyze_conversation(req: AnalyzeConversationRequest) -> AnalyzeConversationResponse:
    """Transcribe a call and run sentiment analysis per segment."""
    logger.info(
        "Analyzing conversation: %s (backend=%s model=%s)", req.audio_path, req.backend, req.model
    )

    async def _do() -> AnalyzeConversationResponse:
        tr = transcribe_helper(
            audio_path=req.audio_path,
            model=req.model,
            backend=req.backend,
            device=req.device,
            language=req.language,
            beam_size=req.beam_size,
            vad=req.vad,
            word_timestamps=req.word_timestamps,
            chunk_length_s=req.chunk_length_s,
            revision=req.revision,
            diarize=req.diarize,
            num_speakers=req.num_speakers,
        )
        segments = tr.get("segments", []) or []
        tr_texts = texts_from_segments(segments)
        results, meta = analyze_smart(
            tr_texts,
            profile="call",
            model_name=req.sentiment_model,
            device=req.device,
            batch_size=16,
            normalize=True,
            return_all_scores=True,
            max_length=None,
            clean=True,
            lexicon_file=req.lexicon_file,
            lexicon_weight=req.lexicon_weight,
        )
        seg_out = _build_segment_sentiments(tr_texts, results, segments)
        return AnalyzeConversationResponse(
            transcript=tr,
            segment_sentiments=seg_out,
            meta=meta,
            timestamp=utc_now_iso(),
        )

    return await run_route("analyze_conversation", _do)


@router.post("/batch_analyze_conversation", response_model=BatchAnalyzeConversationResponse)
async def batch_analyze_conversation(
    req: BatchAnalyzeConversationRequest,
) -> BatchAnalyzeConversationResponse:
    """Analyze sentiment for multiple conversation audio files."""

    async def _do() -> BatchAnalyzeConversationResponse:
        files = resolve_audio_paths(
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
            tr = transcribe_helper(
                audio_path=p,
                model=req.model,
                backend=req.backend,
                device=req.device,
                language=req.language,
                beam_size=req.beam_size,
                vad=req.vad,
                word_timestamps=req.word_timestamps,
                chunk_length_s=req.chunk_length_s,
                revision=req.revision,
                diarize=req.diarize,
                num_speakers=req.num_speakers,
            )
            segments = tr.get("segments", []) or []
            tr_texts = texts_from_segments(segments)
            results, meta = analyze_smart(
                tr_texts,
                profile="call",
                model_name=req.sentiment_model,
                device=req.device,
                batch_size=req.sentiment_batch_size,
                normalize=True,
                return_all_scores=True,
                max_length=None,
                clean=True,
                lexicon_file=req.lexicon_file,
                lexicon_weight=req.lexicon_weight,
            )
            seg_out = _build_segment_sentiments(tr_texts, results, segments)
            return tr, seg_out, meta

        raw = run_batch(files, _worker, workers=req.workers, worker_timeout=req.worker_timeout)
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
