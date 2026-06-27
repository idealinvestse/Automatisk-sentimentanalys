"""Bridge helpers: transcription output → dashboard report state."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from app.nicegui_dashboard.state import DashboardState


def normalize_transcription_to_report(
    transcript_dict: dict[str, Any],
    *,
    call_id: str | None = None,
) -> dict[str, Any]:
    """Convert an ASR transcript payload into a dashboard-compatible report dict."""
    segments = list(transcript_dict.get("segments") or [])

    cid = (
        call_id
        or transcript_dict.get("call_id")
        or transcript_dict.get("id")
    )
    if not cid:
        source = (
            transcript_dict.get("file")
            or transcript_dict.get("filename")
            or transcript_dict.get("source_file")
            or ""
        )
        stem = Path(str(source)).stem if source else ""
        cid = stem or f"TX-{uuid.uuid4().hex[:8].upper()}"

    title = transcript_dict.get("title") or cid
    meta = dict(transcript_dict.get("meta") or {})
    for key, meta_key in (
        ("file", "source_file"),
        ("filename", "source_file"),
        ("duration", "duration_s"),
        ("model", "asr_model"),
        ("backend", "asr_backend"),
    ):
        val = transcript_dict.get(key)
        if val is not None:
            meta.setdefault(meta_key, val)

    sentiment = transcript_dict.get("sentiment_results")
    if not sentiment and segments:
        sentiment = [{"label": "neutral", "score": 0.5} for _ in segments]

    return {
        "call_id": str(cid),
        "title": str(title),
        "meta": meta,
        "segments": segments,
        "sentiment_results": sentiment or [],
        "intent_results": transcript_dict.get("intent_results") or [],
        "results": dict(transcript_dict.get("results") or {}),
        "llm": dict(transcript_dict.get("llm") or {}),
        "risks": dict(transcript_dict.get("risks") or {}),
        "topics": dict(transcript_dict.get("topics") or {}),
        "insights": dict(transcript_dict.get("insights") or {}),
        "processing_time_s": (
            transcript_dict.get("processing_time_s")
            or transcript_dict.get("processing_time")
        ),
        "source": "transcription",
    }


def append_report_to_state(state: DashboardState, report: dict[str, Any]) -> str:
    """Upsert a report into dashboard state and select it. Returns call_id."""
    cid = str(report.get("call_id") or report.get("id") or "")
    if not cid:
        cid = f"TX-{uuid.uuid4().hex[:8].upper()}"
        report["call_id"] = cid

    existing_idx = next(
        (
            i
            for i, r in enumerate(state.reports)
            if str(r.get("call_id") or r.get("id", "")) == cid
        ),
        None,
    )
    if existing_idx is not None:
        state.reports[existing_idx] = report
    else:
        state.reports.append(report)

    state.selected_call_id = cid
    return cid