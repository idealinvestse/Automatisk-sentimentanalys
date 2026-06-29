"""Directory scan + incremental batch processing router (/scan_process)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request

from ...caching import AggregateCache
from ..path_validation import resolve_and_validate_audio_paths
from ...core.serialization import utc_now_iso
from ..batch import file_display_name, run_batch
from ..dependencies import get_cache
from ..helpers import asr_kwargs_from, transcribe_helper
from ..router_errors import run_route
from ..schemas import ScanItem, ScanProcessRequest, ScanProcessResponse
from ..services.conversation import run_batch_analyze_file
from ..transcription_events import JOB_HEADER, get_hub
from ..transcription_jobs import TranscriptionJobRegistry, get_job_registry

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Scan"])


# ---------------------------------------------------------------------------
# State file helpers
# ---------------------------------------------------------------------------


def _load_state(path: str | None) -> dict[str, Any]:
    """Load processing state from a JSON file (returns empty state on any error)."""
    if not path or not os.path.isfile(path):
        return {"processed": {}}
    try:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and "processed" in obj and isinstance(obj["processed"], dict):
            return obj
    except (OSError, json.JSONDecodeError):
        pass
    return {"processed": {}}


def _save_state(path: str | None, state: dict[str, Any]) -> None:
    """Persist processing state to a JSON file (silently no-ops if path is None)."""
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _chunk(lst: list, size: int) -> list[list]:
    """Divide *lst* into sub-lists of at most *size* elements."""
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def _run_scan_process(
    req: ScanProcessRequest,
    job_id: str | None,
    hub: Any,
    cache: AggregateCache | None,
    registry: TranscriptionJobRegistry | None = None,
) -> ScanProcessResponse:
    """Synchronous scan body — executed in a worker thread from the async handler."""
    files = resolve_and_validate_audio_paths(
        audio_paths=None,
        directory=req.directory,
        pattern=req.pattern,
        recursive=req.recursive,
    )

    state = _load_state(req.state_file)
    processed: dict[str, Any] = state.get("processed", {})

    new_files: list[str] = []
    skipped = 0
    for p in files:
        try:
            mtime = os.path.getmtime(p)
        except OSError:
            continue
        info = processed.get(p)
        if info and isinstance(info, dict) and float(info.get("mtime", 0.0)) >= float(mtime):
            skipped += 1
        else:
            new_files.append(p)

    if req.max_files:
        new_files = new_files[: req.max_files]

    logger.info(
        "scan_process: %d file(s) to process, %d already up-to-date",
        len(new_files),
        skipped,
    )

    def _do_transcribe(p: str) -> dict[str, Any]:
        return transcribe_helper(**asr_kwargs_from(req, audio_path=p))

    def _do_analyze(p: str) -> dict[str, Any]:
        tr, seg_out, meta, pipe_results = run_batch_analyze_file(req, p, cache=cache)
        seg_dicts = [s.model_dump() for s in seg_out]
        out: dict[str, Any] = {"transcript": tr, "segment_sentiments": seg_dicts, "meta": meta}
        if pipe_results is not None:
            out["pipeline_results"] = pipe_results
        return out

    worker_fn = _do_transcribe if req.operation == "transcribe" else _do_analyze

    batches = _chunk(new_files, req.batch_size)
    items: list[ScanItem] = []
    ok = failed = 0
    total = len(new_files)
    processed_count = 0

    hub.log(job_id=job_id, level="INFO", msg=f"scan_process startar – {total} nya filer")
    hub.status(job_id=job_id, is_running=True, total=total, processed=0)

    cancelled = False
    for bidx, batch in enumerate(batches):
        if registry and registry.is_cancelled(job_id):
            cancelled = True
            hub.log(job_id=job_id, level="WARNING", msg="scan_process avbruten mellan batchar")
            break
        def _on_complete(
            path: str,
            result: dict[str, Any] | None,
            error: Exception | None,
            done: int,
            batch_total: int,
        ) -> None:
            nonlocal processed_count
            processed_count += 1
            fname = file_display_name(path)
            progress = round(processed_count / max(1, total), 2)
            hub.progress(
                job_id=job_id,
                processed=processed_count,
                total=total,
                current_file=fname,
                progress=progress,
            )
            if error is None:
                hub.log(job_id=job_id, level="INFO", msg=f"[scan_process] Klart: {fname}", file=fname)
            else:
                hub.log(job_id=job_id, level="ERROR", msg=str(error), file=fname)

        raw = run_batch(
            batch,
            worker_fn,
            workers=req.workers,
            worker_timeout=req.worker_timeout,
            on_file_complete=_on_complete,
            should_cancel=lambda: bool(registry and registry.is_cancelled(job_id)),
        )

        for path, result, error in raw:
            if error is None:
                items.append(ScanItem(file=path, ok=True, data=result, batch_index=bidx))
                ok += 1
                with contextlib.suppress(OSError):
                    processed[path] = {
                        "mtime": os.path.getmtime(path),
                        "when": utc_now_iso(trim_microseconds=False),
                    }
            else:
                logger.error("scan_process failed for %s: %s", path, error, exc_info=True)
                items.append(
                    ScanItem(file=path, ok=False, error=str(error), batch_index=bidx)
                )
                failed += 1

        state["processed"] = processed
        _save_state(req.state_file, state)
        logger.debug("scan_process: batch %d/%d done (ok=%d failed=%d)", bidx + 1, len(batches), ok, failed)

    if skipped:
        hub.log(job_id=job_id, level="INFO", msg=f"Hoppade över {skipped} redan bearbetade filer")
    end_msg = (
        f"scan_process avbruten – {ok} ok, {failed} fel"
        if cancelled
        else f"scan_process slutförd – {ok} ok, {failed} fel"
    )
    hub.log(job_id=job_id, level="INFO", msg=end_msg)
    hub.done(job_id=job_id, ok=ok, failed=failed)
    hub.status(job_id=job_id, is_running=False)
    if registry and job_id:
        registry.complete(job_id, status="cancelled" if cancelled else "completed")

    return ScanProcessResponse(
        items=items,
        ok=ok,
        failed=failed,
        total=len(new_files),
        skipped=skipped,
        timestamp=utc_now_iso(),
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/scan_process", response_model=ScanProcessResponse)
async def scan_process(
    req: ScanProcessRequest,
    request: Request,
    cache: Annotated[AggregateCache, Depends(get_cache)],
) -> ScanProcessResponse:
    """Scan a directory and process new/changed audio files incrementally.

    Tracks processed files via an optional state JSON file so that re-runs
    only process files that are new or have been modified since the last run.

    **Race-condition fix:** State is written to disk after *each batch*
    (not only at the end) so that progress is preserved if the process is
    interrupted mid-run.

    Args:
        req: Scan parameters including directory, pattern, batch size,
             workers, and ASR/sentiment settings.

    Returns:
        Per-file results, counts, and a UTC timestamp.
    """
    logger.info("scan_process: directory=%s pattern=%s operation=%s", req.directory, req.pattern, req.operation)
    job_id = request.headers.get(JOB_HEADER)
    hub = get_hub(request.app)
    registry = get_job_registry(request.app)
    if job_id:
        registry.register(job_id, "scan_process", directory=req.directory, operation=req.operation)

    async def _do() -> ScanProcessResponse:
        return await asyncio.to_thread(_run_scan_process, req, job_id, hub, cache, registry)

    return await run_route("scan_process", _do)
