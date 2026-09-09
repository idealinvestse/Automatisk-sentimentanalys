"""Shared LLM transcript helpers (used by Mistral and Groq analyzers)."""

from __future__ import annotations

import hashlib
import json
from typing import Any, cast

from ..core.models import Segment


def build_role_labeled_transcript(
    segments: list[dict[str, Any]] | list[Segment],
    role_map: dict[str, str] | None = None,
) -> str:
    """Turn segments into a clean, role-aware transcript for LLM prompts."""
    lines: list[str] = []
    if role_map and isinstance(role_map.get("roles"), dict):
        role_map = cast(dict[str, str], role_map["roles"])
    for index, seg in enumerate(segments):
        if isinstance(seg, dict):
            text = str(seg.get("text", "")).strip()
            speaker = seg.get("speaker") or seg.get("speaker_label") or "UNKNOWN"
            start = seg.get("start")
            end = seg.get("end")
            segment_id = seg.get("segment_id", seg.get("id", index))
        else:
            text = getattr(seg, "text", "").strip()
            speaker = getattr(seg, "speaker", None) or "UNKNOWN"
            start = getattr(seg, "start", None)
            end = getattr(seg, "end", None)
            segment_id = getattr(seg, "segment_id", index)
        if not text:
            continue

        role = "UNKNOWN"
        if role_map and speaker in role_map:
            role = str(role_map[speaker]).upper()
        elif speaker and "agent" in str(speaker).lower():
            role = "AGENT"
        elif speaker and "customer" in str(speaker).lower():
            role = "CUSTOMER"

        prefix = f"[{role}]" if role != "UNKNOWN" else f"[{speaker}]"
        timing = ""
        if start is not None or end is not None:
            timing = f" start={start if start is not None else '?'} end={end if end is not None else '?'}"
        lines.append(f"{prefix} {text} [segment={segment_id}{timing}]")

    return "\n".join(lines)


def make_transcript_hash(transcript: str, role_map: dict[str, str] | None) -> str:
    """Short stable hash for LLM response caching."""
    payload = transcript + json.dumps(role_map or {}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
