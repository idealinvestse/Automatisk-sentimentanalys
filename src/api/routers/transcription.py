"""ASR transcription routers (/transcribe, /batch_transcribe, /upload)."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile

from ...core.serialization import utc_now_iso
from ..batch import file_display_name, run_batch
from ..helpers import asr_kwargs_from, transcribe_helper
from ..path_validation import resolve_and_validate_audio_paths
from ..router_errors import run_route
from ..schemas import (
    BatchTranscribeItem,
    BatchTranscribeRequest,
    BatchTranscribeResponse,
    TranscribeJobCancelResponse,
    TranscribeJobListResponse,
    TranscribeJobStatus,
    TranscribeRequest,
    TranscribeResponse,
    UploadResponse,
)
from ..transcription_events import JOB_HEADER, get_hub
from ..transcription_jobs import TranscriptionJob, get_job_registry

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Transcription"])


def _job_id(request: Request) -> str | None:
    return request.headers.get(JOB_HEADER)


def _register_job(request: Request, job_id: str | None, kind: str) -> TranscriptionJob | None:
    if not job_id:
        return None
    return get_job_registry(request.app).register(job_id, kind)


def _cancel_check(request: Request, job_id: str | None) -> bool:
    return get_job_registry(request.app).is_cancelled(job_id)


def _cleanup_old_uploads(upload_dir: Path, retention_days: int) -> None:
    """Clean up uploaded files older than retention_days."""
    if not upload_dir.exists():
        return

    cutoff_time = datetime.now() - timedelta(days=retention_days)
    cleaned = 0

    try:
        for file_path in upload_dir.iterdir():
            if file_path.is_file():
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_mtime < cutoff_time:
                    file_path.unlink()
                    cleaned += 1
        if cleaned > 0:
            logger.info("Cleaned up %d old upload(s) from %s", cleaned, upload_dir)
    except Exception as e:
        logger.warning("Failed to clean up old uploads: %s", e)


@router.get("/transcription/jobs", response_model=TranscribeJobListResponse)
async def list_transcription_jobs(request: Request, limit: int = 20) -> TranscribeJobListResponse:
    """List recent transcription jobs (newest first)."""
    jobs = get_job_registry(request.app).list_jobs(limit=limit)
    return TranscribeJobListResponse(jobs=jobs, timestamp=utc_now_iso())


@router.get("/transcription/jobs/{job_id}", response_model=TranscribeJobStatus)
async def get_transcription_job(job_id: str, request: Request) -> TranscribeJobStatus:
    """Return status snapshot for a transcription job."""
    job = get_job_registry(request.app).get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return TranscribeJobStatus(**job.to_dict())


@router.post("/transcription/jobs/{job_id}/cancel", response_model=TranscribeJobCancelResponse)
async def cancel_transcription_job(job_id: str, request: Request) -> TranscribeJobCancelResponse:
    """Request cancellation of a running transcription job."""
    registry = get_job_registry(request.app)
    if not registry.cancel(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    hub = get_hub(request.app)
    hub.log(job_id=job_id, level="WARNING", msg="Jobb avbrutet av klient")
    hub.status(job_id=job_id, is_running=False)
    return TranscribeJobCancelResponse(job_id=job_id, cancelled=True, timestamp=utc_now_iso())


@router.post("/upload", response_model=UploadResponse)
async def upload_audio_file(file: UploadFile, request: Request) -> UploadResponse:
    """Upload an audio file to the server for transcription.

    The file is saved under API_MEDIA_ROOT/uploads/ with a unique filename.
    Returns the validated server-side path that can be used with POST /transcribe.

    Supports common audio formats: wav, mp3, m4a, flac, ogg, webm.
    Maximum file size: API_MAX_UPLOAD_SIZE_MB (default 200 MB).
    Old files are cleaned up after API_UPLOAD_RETENTION_DAYS (default 7 days).
    """
    from ..settings import get_api_settings

    settings = get_api_settings()

    # Validate media root is configured
    if not settings.media_root:
        raise HTTPException(
            status_code=500,
            detail="Server not configured for file uploads (API_MEDIA_ROOT not set)",
        )

    # Validate file extension
    allowed_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".opus"}
    file_ext = Path(file.filename or "").suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_ext}. Allowed: {', '.join(allowed_extensions)}",
        )

    # Create uploads directory if it doesn't exist
    upload_dir = Path(settings.media_root) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Clean up old uploads (best-effort, runs on upload to avoid startup complexity)
    _cleanup_old_uploads(upload_dir, settings.upload_retention_days)

    # Generate unique filename (uuid4 + original name)
    unique_id = uuid.uuid4().hex[:12]
    safe_filename = f"{unique_id}_{file.filename}"
    safe_filename = safe_filename.replace(" ", "_").replace("/", "_")
    file_path = upload_dir / safe_filename

    # Save file with streaming read and size check
    max_size_bytes = settings.max_upload_size_mb * 1024 * 1024
    total_bytes = 0
    chunk_size = 64 * 1024  # 64 KB chunks

    try:
        with open(file_path, "wb") as f:
            while chunk := await file.read(chunk_size):
                total_bytes += len(chunk)
                if total_bytes > max_size_bytes:
                    f.close()
                    file_path.unlink()  # Clean up partial file
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large (max {settings.max_upload_size_mb} MB)",
                    )
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to save uploaded file: %s", e)
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Validate the saved file exists and is under media root
    try:
        validated_path = resolve_and_validate_audio_paths([str(file_path)])[0]
    except ValueError as e:
        # Clean up if validation fails
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=400, detail=f"File validation failed: {str(e)}")

    return UploadResponse(
        audio_path=validated_path,
        filename=file.filename or "unknown",
        size_bytes=total_bytes,
        timestamp=utc_now_iso(),
    )


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(req: TranscribeRequest, request: Request) -> TranscribeResponse:
    """Transcribe an audio file using ASR."""
    job_id = _job_id(request)
    hub = get_hub(request.app)
    registry = get_job_registry(request.app)
    _register_job(request, job_id, "transcribe")
    fname = file_display_name(req.audio_path)
    logger.info("Transcribing %s (backend=%s model=%s)", req.audio_path, req.backend, req.model)
    hub.log(job_id=job_id, level="INFO", msg=f"Startar transkribering: {fname}", file=fname)
    hub.progress(job_id=job_id, processed=0, total=1, current_file=fname, progress=0.0)
    hub.status(job_id=job_id, is_running=True)

    async def _do() -> TranscribeResponse:
        try:
            if _cancel_check(request, job_id):
                raise asyncio.CancelledError("Job cancelled")
            tr = await asyncio.to_thread(
                transcribe_helper,
                **asr_kwargs_from(req, audio_path=req.audio_path, preprocess=req.preprocess),
            )
            if _cancel_check(request, job_id):
                raise asyncio.CancelledError("Job cancelled")
            n_seg = len(tr.get("segments") or [])
            hub.log(
                job_id=job_id,
                level="INFO",
                msg=f"Klart: {fname} – {n_seg} segment",
                file=fname,
            )
            hub.progress(job_id=job_id, processed=1, total=1, current_file=fname, progress=1.0)
            hub.done(job_id=job_id, ok=1, failed=0)
            if job_id:
                registry.complete(job_id, status="completed")
            return TranscribeResponse(transcript=tr, timestamp=utc_now_iso())
        except asyncio.CancelledError:
            hub.log(job_id=job_id, level="WARNING", msg="Avbruten", file=fname)
            hub.done(job_id=job_id, ok=0, failed=0)
            if job_id:
                registry.complete(job_id, status="cancelled")
            raise HTTPException(status_code=409, detail="Job cancelled") from None
        except Exception as err:
            hub.log(job_id=job_id, level="ERROR", msg=str(err), file=fname)
            hub.done(job_id=job_id, ok=0, failed=1)
            if job_id:
                registry.complete(job_id, status="failed")
            raise
        finally:
            hub.status(job_id=job_id, is_running=False)

    return await run_route("transcribe", _do)


@router.post("/batch_transcribe", response_model=BatchTranscribeResponse)
async def batch_transcribe(
    req: BatchTranscribeRequest, request: Request
) -> BatchTranscribeResponse:
    """Transcribe multiple audio files, optionally in parallel."""
    job_id = _job_id(request)
    hub = get_hub(request.app)
    registry = get_job_registry(request.app)
    _register_job(request, job_id, "batch_transcribe")

    async def _do() -> BatchTranscribeResponse:
        files = resolve_and_validate_audio_paths(
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
            return transcribe_helper(**asr_kwargs_from(req, audio_path=p))

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

        raw = await asyncio.to_thread(
            run_batch,
            files,
            _worker,
            workers=req.workers,
            worker_timeout=req.worker_timeout,
            on_file_complete=_on_complete,
            should_cancel=lambda: registry.is_cancelled(job_id),
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
        cancelled = registry.is_cancelled(job_id)
        msg = (
            f"Batch avbruten – {ok} ok, {failed} fel"
            if cancelled
            else f"Batch slutförd – {ok} ok, {failed} fel"
        )
        hub.log(job_id=job_id, level="INFO", msg=msg)
        hub.done(job_id=job_id, ok=ok, failed=failed)
        hub.status(job_id=job_id, is_running=False)
        if job_id:
            registry.complete(job_id, status="cancelled" if cancelled else "completed")
        return BatchTranscribeResponse(
            items=items, ok=ok, failed=failed, total=len(files), timestamp=utc_now_iso()
        )

    return await run_route("batch_transcribe", _do)
