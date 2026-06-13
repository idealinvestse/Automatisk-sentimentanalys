"""Virtual list helpers for large call transcripts.

Fas 6.1 – docs/MIGRATION_TO_NICEGUI_PLAN.md (virtualisering av transkript)
"""

from __future__ import annotations

from typing import Any

# Approximate row heights (px) for spacer-based virtual scrolling
TRANSCRIPT_ROW_HEIGHT = 72
TIMELINE_ROW_HEIGHT = 52
VIRTUALIZE_THRESHOLD = 40
DEFAULT_WINDOW_SIZE = 22
OVERSCAN_ROWS = 4


def should_virtualize(segment_count: int) -> bool:
    """Use virtual scrolling when segment count exceeds threshold."""
    return segment_count >= VIRTUALIZE_THRESHOLD


def filter_segments_with_index(
    enriched: list[dict[str, Any]],
    search: str,
) -> list[tuple[int, dict[str, Any]]]:
    """Return (original_index, segment) pairs matching search query."""
    query = (search or "").strip().lower()
    if not query:
        return list(enumerate(enriched))
    return [
        (i, seg)
        for i, seg in enumerate(enriched)
        if query in str(seg.get("text", "")).lower()
        or query in str(seg.get("speaker", "")).lower()
    ]


def compute_visible_range(
    scroll_top: float,
    container_height: float,
    total_items: int,
    row_height: float = TRANSCRIPT_ROW_HEIGHT,
    *,
    overscan: int = OVERSCAN_ROWS,
) -> tuple[int, int]:
    """Return (start_idx, end_idx) for the visible window (end exclusive)."""
    if total_items <= 0:
        return 0, 0
    start = max(0, int(scroll_top / row_height) - overscan)
    visible_rows = max(1, int(container_height / row_height) + 1)
    end = min(total_items, start + visible_rows + overscan * 2)
    if end <= start:
        end = min(total_items, start + DEFAULT_WINDOW_SIZE)
    return start, end


def window_around_index(
    center_idx: int,
    total_items: int,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> tuple[int, int]:
    """Return window centered on index (for timeline / jump-to-segment)."""
    if total_items <= 0:
        return 0, 0
    half = window_size // 2
    start = max(0, center_idx - half)
    end = min(total_items, start + window_size)
    start = max(0, end - window_size)
    return start, end


def scroll_pixels_for_index(index: int, row_height: float = TRANSCRIPT_ROW_HEIGHT) -> float:
    """Pixel offset to scroll a row into view."""
    return max(0.0, float(index) * row_height)


def make_synthetic_segments(count: int, *, base_text: str = "Segment") -> list[dict[str, Any]]:
    """Generate synthetic segments for tests / perf checks."""
    segments: list[dict[str, Any]] = []
    for i in range(count):
        speaker = "Agent" if i % 2 == 0 else "Kund"
        segments.append(
            {
                "start": float(i * 12),
                "end": float(i * 12 + 10),
                "text": f"{base_text} {i + 1}: exempelrad för virtualiseringstest.",
                "speaker": speaker,
            }
        )
    return segments