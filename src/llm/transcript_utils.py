"""Shared LLM transcript helpers (used by Mistral and Groq analyzers)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ..core.models import Segment


def build_role_labeled_transcript(
    segments: list[dict[str, Any]] | list[Segment],
    role_map: dict[str, str] | None = None,
) -> str:
    """Turn segments into a clean, role-aware transcript for LLM prompts."""
    lines: list[str] = []
    for seg in segments:
        if isinstance(seg, dict):
            text = seg.get("text", "").strip()
            speaker = seg.get("speaker") or seg.get("speaker_label") or "UNKNOWN"
        else:
            text = getattr(seg, "text", "").strip()
            speaker = getattr(seg, "speaker", None) or "UNKNOWN"

        role = "UNKNOWN"
        if role_map and speaker in role_map:
            role = role_map[speaker].upper()
        elif speaker and "agent" in str(speaker).lower():
            role = "AGENT"
        elif speaker and "customer" in str(speaker).lower():
            role = "CUSTOMER"

        prefix = f"[{role}]" if role != "UNKNOWN" else f"[{speaker}]"
        lines.append(f"{prefix} {text}")

    return "\n".join(lines)


def make_transcript_hash(transcript: str, role_map: dict[str, str] | None) -> str:
    """Short stable hash for LLM response caching."""
    payload = transcript + json.dumps(role_map or {}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
