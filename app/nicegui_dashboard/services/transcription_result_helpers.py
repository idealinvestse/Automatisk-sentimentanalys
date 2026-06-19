"""Helpers for ad-hoc transcription result display and export."""

from __future__ import annotations

import csv
import io
import json
from typing import Any


def fmt_timestamp(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    total = max(0, int(seconds))
    mins, secs = divmod(total, 60)
    return f"{mins:02d}:{secs:02d}"


def confidence_color(conf: float | None) -> str:
    if conf is None:
        return "text-grey"
    if conf < 0.60:
        return "text-negative"
    if conf < 0.80:
        return "text-warning"
    return "text-positive"


def confidence_pct(conf: float | None) -> str:
    if conf is None:
        return "—"
    return f"{conf * 100:.0f}%"


def speaker_label(speaker: str | int | None) -> str:
    if speaker is None:
        return "—"
    raw = str(speaker).strip()
    if not raw:
        return "—"
    upper = raw.upper()
    if upper in {"SPEAKER_00", "0"}:
        return "Agent"
    if upper in {"SPEAKER_01", "1"}:
        return "Kund"
    if upper.startswith("SPEAKER_"):
        idx = upper.replace("SPEAKER_", "")
        if idx.isdigit():
            return "Agent" if int(idx) % 2 == 0 else "Kund"
    return raw


def speaker_chip_color(speaker: str | int | None) -> str:
    label = speaker_label(speaker)
    if label == "Agent":
        return "primary"
    if label == "Kund":
        return "secondary"
    return "grey"


def transcript_full_text(transcript: dict[str, Any]) -> str:
    segments = transcript.get("segments") or []
    return " ".join(str(s.get("text", "")).strip() for s in segments).strip()


def segments_table_rows(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, seg in enumerate(transcript.get("segments") or []):
        conf = seg.get("confidence")
        if conf is None:
            conf = seg.get("avg_confidence")
        rows.append(
            {
                "idx": i + 1,
                "time": f"{fmt_timestamp(seg.get('start'))} – {fmt_timestamp(seg.get('end'))}",
                "speaker": speaker_label(seg.get("speaker")),
                "text": seg.get("text", ""),
                "confidence": confidence_pct(conf),
                "confidence_raw": conf,
                "low_confidence": bool(seg.get("low_confidence")),
                "warning": "⚠ Låg" if seg.get("low_confidence") else "",
            }
        )
    return rows


def transcript_llm_enhanced(transcript: dict[str, Any], meta: dict[str, Any] | None = None) -> bool:
    meta = meta or {}
    flags = (
        transcript.get("llm_enhanced"),
        transcript.get("mistral_enhanced"),
        meta.get("llm_enhanced"),
        meta.get("mistral_used"),
        meta.get("deep_analysis"),
    )
    return any(bool(f) for f in flags)


def segments_to_csv_bytes(transcript: dict[str, Any]) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["index", "start", "end", "speaker", "text", "confidence", "low_confidence"])
    for i, seg in enumerate(transcript.get("segments") or []):
        conf = seg.get("confidence")
        if conf is None:
            conf = seg.get("avg_confidence")
        writer.writerow(
            [
                i + 1,
                seg.get("start"),
                seg.get("end"),
                speaker_label(seg.get("speaker")),
                seg.get("text", ""),
                conf,
                seg.get("low_confidence", False),
            ]
        )
    return buf.getvalue().encode("utf-8-sig")


def transcript_to_json_bytes(transcript: dict[str, Any]) -> bytes:
    return json.dumps(transcript, indent=2, ensure_ascii=False).encode("utf-8")