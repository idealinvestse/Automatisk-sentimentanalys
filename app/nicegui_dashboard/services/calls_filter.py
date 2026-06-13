"""Calls table search and pagination helpers.

Post-migration UX – docs/MIGRATION_TO_NICEGUI_PLAN.md (Fas 6.1)
"""

from __future__ import annotations

from typing import Any

DEFAULT_TABLE_PAGE_SIZE = 20


def search_table_reports(
    reports: list[dict[str, Any]],
    query: str,
) -> list[dict[str, Any]]:
    """Filter reports by call_id, title, agent, or any segment text."""
    q = (query or "").strip().lower()
    if not q:
        return list(reports)

    matched: list[dict[str, Any]] = []
    for report in reports:
        call_id = str(report.get("call_id") or report.get("id", "")).lower()
        title = str(report.get("title", "")).lower()
        agent = str((report.get("meta") or {}).get("agent", "")).lower()
        segment_text = " ".join(
            str(seg.get("text", "")) for seg in (report.get("segments") or [])
        ).lower()
        if q in f"{call_id} {title} {agent} {segment_text}":
            matched.append(report)
    return matched


def paginate_items(
    items: list[Any],
    page: int,
    page_size: int = DEFAULT_TABLE_PAGE_SIZE,
) -> tuple[list[Any], int, int]:
    """Return (page_slice, total_pages, total_count)."""
    total = len(items)
    if total == 0:
        return [], 1, 0
    total_pages = max(1, (total + page_size - 1) // page_size)
    safe_page = max(1, min(page, total_pages))
    start = (safe_page - 1) * page_size
    return items[start : start + page_size], total_pages, total