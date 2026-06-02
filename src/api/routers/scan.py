"""Directory scan + incremental batch processing router (/scan_process)."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException

from ...core.audio import resolve_audio_paths
from ...core.serialization import map_results_to_segment_dicts, texts_from_segments, utc_now_iso
from ...lexicon import blend_results_with_lexicon
from ...sentiment import analyze_smart
from ..batch import run_batch
from ..helpers import transcribe_helper
from ..schemas import ScanItem, ScanProcessRequest, ScanProcessResponse

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


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/scan_process", response_model=ScanProcessResponse)
async def scan_process(req: ScanProcessRequest) -> ScanProcessResponse:
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

    # --- Resolve files -------------------------------------------------------
    files = resolve_audio_paths(
        audio_paths=None,
        directory=req.directory,
        pattern=req.pattern,
        recursive=req.recursive,
    )

    # --- Load state and filter to new/modified files -------------------------
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

    # --- Worker functions ----------------------------------------------------

    def _do_transcribe(p: str) -> dict[str, Any]:
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

    def _do_analyze(p: str) -> dict[str, Any]:
        tr = _do_transcribe(p)
        segments = tr.get("segments", []) or []
        texts = texts_from_segments(segments)
        results, meta = analyze_smart(
            texts,
            profile="call",
            model_name=req.sentiment_model,
            device=req.device,
            batch_size=req.sentiment_batch_size,
            normalize=True,
            return_all_scores=True,
            max_length=None,
            clean=True,
        )
        results = blend_results_with_lexicon(texts, results, req.lexicon_file, req.lexicon_weight)
        seg_dicts = map_results_to_segment_dicts(texts, results, segments)
        return {"transcript": tr, "segment_sentiments": seg_dicts, "meta": meta}

    worker_fn = _do_transcribe if req.operation == "transcribe" else _do_analyze

    # --- Process in batches, saving state after each batch -------------------
    batches = _chunk(new_files, req.batch_size)
    items: list[ScanItem] = []
    ok = failed = 0

    for bidx, batch in enumerate(batches):
        raw = run_batch(batch, worker_fn, workers=req.workers, worker_timeout=req.worker_timeout)

        for path, result, error in raw:
            if error is None:
                items.append(ScanItem(file=path, ok=True, data=result, batch_index=bidx))
                ok += 1
                # Record successful processing with current mtime
                try:
                    processed[path] = {
                        "mtime": os.path.getmtime(path),
                        "when": utc_now_iso(trim_microseconds=False),
                    }
                except OSError:
                    pass
            else:
                logger.error("scan_process failed for %s: %s", path, error, exc_info=True)
                items.append(
                    ScanItem(file=path, ok=False, error=str(error), batch_index=bidx)
                )
                failed += 1

        # Persist state after every batch so progress survives interruption
        state["processed"] = processed
        _save_state(req.state_file, state)
        logger.debug("scan_process: batch %d/%d done (ok=%d failed=%d)", bidx + 1, len(batches), ok, failed)

    return ScanProcessResponse(
        items=items,
        ok=ok,
        failed=failed,
        total=len(new_files),
        skipped=skipped,
        timestamp=utc_now_iso(),
    )
