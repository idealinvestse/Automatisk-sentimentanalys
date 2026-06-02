"""ASR transcription routers (/transcribe, /batch_transcribe)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from ...core.audio import resolve_audio_paths
from ...core.serialization import utc_now_iso
from ..batch import run_batch
from ..helpers import transcribe_helper
from ..schemas import (
    BatchTranscribeItem,
    BatchTranscribeRequest,
    BatchTranscribeResponse,
    TranscribeRequest,
    TranscribeResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Transcription"])


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(req: TranscribeRequest) -> TranscribeResponse:
    """Transcribe an audio file using ASR.

    Uses a cached transcriber instance so that model weights are only loaded
    once per unique (backend, model, device) combination.

    Returns:
        Transcription result dict and a UTC timestamp.
    """
    logger.info("Transcribing %s (backend=%s model=%s)", req.audio_path, req.backend, req.model)
    try:
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
    except Exception as e:
        logger.error("Transcription failed for %s: %s", req.audio_path, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}") from e

    return TranscribeResponse(transcript=tr, timestamp=utc_now_iso())


@router.post("/batch_transcribe", response_model=BatchTranscribeResponse)
async def batch_transcribe(req: BatchTranscribeRequest) -> BatchTranscribeResponse:
    """Transcribe multiple audio files, optionally in parallel.

    Files can be specified as explicit paths, directories, or glob patterns.

    Returns:
        Per-file results with ok/failed counts and a UTC timestamp.
    """
    files = resolve_audio_paths(
        audio_paths=req.audio_paths,
        directory=req.directory,
        pattern=req.glob,
        recursive=req.recursive,
        limit=req.limit,
    )
    logger.info("Batch transcribing %d file(s) with %d worker(s)", len(files), req.workers)

    def _worker(p: str) -> dict:
        return transcribe_helper(
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

    raw = run_batch(files, _worker, workers=req.workers, worker_timeout=req.worker_timeout)

    items: list[BatchTranscribeItem] = []
    ok = failed = 0
    for path, result, error in raw:
        if error is None:
            items.append(BatchTranscribeItem(file=path, transcript=result))
            ok += 1
        else:
            items.append(BatchTranscribeItem(file=path, error=str(error)))
            failed += 1

    return BatchTranscribeResponse(
        items=items, ok=ok, failed=failed, total=len(files), timestamp=utc_now_iso()
    )
