"""ASR transcription routers (/transcribe, /batch_transcribe)."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Request

from ...core.audio import resolve_audio_paths
from ...core.serialization import utc_now_iso
from ..batch import file_display_name, run_batch
from ..helpers import transcribe_helper
from ..router_errors import run_route
from ..schemas import (
    BatchTranscribeItem,
    BatchTranscribeRequest,
    BatchTranscribeResponse,
    TranscribeRequest,
    TranscribeResponse,
)
from ..transcription_events import JOB_HEADER, get_hub

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Transcription"])


def _job_id(request: Request) -> str | None:
    return request.headers.get(JOB_HEADER)


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(req: TranscribeRequest, request: Request) -> TranscribeResponse:
    """Transcribe an audio file using ASR."""
    job_id = _job_id(request)
    hub = get_hub(request.app)
    fname = file_display_name(req.audio_path)
    logger.info("Transcribing %s (backend=%s model=%s)", req.audio_path, req.backend, req.model)
    hub.log(job_id=job_id, level="INFO", msg=f"Startar transkribering: {fname}", file=fname)
    hub.progress(job_id=job_id, processed=0, total=1, current_file=fname, progress=0.0)
    hub.status(job_id=job_id, is_running=True)

    async def _do() -> TranscribeResponse:
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
                hotwords=getattr(req, "hotwords", None),
                initial_prompt=getattr(req, "initial_prompt", None),
                preprocess=getattr(req, "preprocess", False),
            )
            n_seg = len(tr.get("segments") or [])
            hub.log(
                job_id=job_id,
                level="INFO",
                msg=f"Klart: {fname} – {n_seg} segment",
                file=fname,
            )
            hub.progress(job_id=job_id, processed=1, total=1, current_file=fname, progress=1.0)
            hub.done(job_id=job_id, ok=1, failed=0)
            return TranscribeResponse(transcript=tr, timestamp=utc_now_iso())
        except Exception as err:
            hub.log(job_id=job_id, level="ERROR", msg=str(err), file=fname)
            hub.done(job_id=job_id, ok=0, failed=1)
            raise
        finally:
            hub.status(job_id=job_id, is_running=False)

    return await run_route("transcribe", _do)


@router.post("/batch_transcribe", response_model=BatchTranscribeResponse)
async def batch_transcribe(req: BatchTranscribeRequest, request: Request) -> BatchTranscribeResponse:
    """Transcribe multiple audio files, optionally in parallel."""
    job_id = _job_id(request)
    hub = get_hub(request.app)

    async def _do() -> BatchTranscribeResponse:
        files = resolve_audio_paths(
            audio_paths=req.audio_paths,
            directory=req.directory,
            pattern=req.glob,
            recursive=req.recursive,
            limit=req.limit,
        )
        total = len(files)
        logger.info("Batch transcribing %d file(s) with %d worker(s)", total, req.workers)
        hub.log(job_id=job_id, level="INFO", msg=f"Batch startar – {total} filer")
        hub.status(job_id=job_id, is_running=True, total=total, processed=0)

        def _worker(p: str) -> dict:
            fname = file_display_name(p)
            hub.log(job_id=job_id, level="INFO", msg=f"Bearbetar {fname}...", file=fname)
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
                hotwords=getattr(req, "hotwords", None),
                initial_prompt=getattr(req, "initial_prompt", None),
                preprocess=getattr(req, "preprocess", False),
            )

        def _on_complete(
            path: str,
            result: dict | None,
            error: Exception | None,
            processed: int,
            batch_total: int,
        ) -> None:
            fname = file_display_name(path)
            progress = round(processed / max(1, batch_total), 2)
            hub.progress(
                job_id=job_id,
                processed=processed,
                total=batch_total,
                current_file=fname,
                progress=progress,
            )
            if error is None and result is not None:
                n_seg = len(result.get("segments") or [])
                hub.log(
                    job_id=job_id,
                    level="INFO",
                    msg=f"Klart: {fname} – {n_seg} segment",
                    file=fname,
                )
            elif error is not None:
                hub.log(job_id=job_id, level="ERROR", msg=str(error), file=fname)

        raw = run_batch(
            files,
            _worker,
            workers=req.workers,
            worker_timeout=req.worker_timeout,
            on_file_complete=_on_complete,
        )
        items: list[BatchTranscribeItem] = []
        ok = failed = 0
        for path, result, error in raw:
            if error is None:
                items.append(BatchTranscribeItem(file=path, transcript=result))
                ok += 1
            else:
                items.append(BatchTranscribeItem(file=path, error=str(error)))
                failed += 1
        hub.log(job_id=job_id, level="INFO", msg=f"Batch slutförd – {ok} ok, {failed} fel")
        hub.done(job_id=job_id, ok=ok, failed=failed)
        hub.status(job_id=job_id, is_running=False)
        return BatchTranscribeResponse(
            items=items, ok=ok, failed=failed, total=len(files), timestamp=utc_now_iso()
        )

    return await run_route("batch_transcribe", _do)