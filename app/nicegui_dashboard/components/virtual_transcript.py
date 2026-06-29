"""Virtualized transcript and timeline rendering for Call Detail.

Fas 6.1 – docs/archive/MIGRATION_TO_NICEGUI_PLAN.md (virtualisering av transkript)
Uses ui.scroll_area + spacer-based window rendering.
"""

from __future__ import annotations

import html
from collections.abc import Callable
from typing import Any

from nicegui import ui
from nicegui.elements.label import Label
from nicegui.events import ScrollEventArguments

from app.nicegui_dashboard.services.transcript_virtualizer import (
    TIMELINE_ROW_HEIGHT,
    TRANSCRIPT_ROW_HEIGHT,
    compute_visible_range,
    filter_segments_with_index,
    highlight_search_text,
    scroll_pixels_for_index,
    should_virtualize,
    window_around_index,
)


def _segment_time_label(seg: dict[str, Any]) -> str:
    start = seg.get("start")
    end = seg.get("end")
    if start is not None and end is not None:
        return f"[{start:.0f}s–{end:.0f}s]"
    return f"[#{seg.get('turn_idx', 0)}]"


def _sentiment_css(seg: dict[str, Any]) -> str:
    classes: list[str] = []
    if seg.get("is_negative_peak"):
        classes.append("segment-negative-peak")
    sent = str(seg.get("sentiment_label", "neutral")).lower()
    if "neg" in sent:
        classes.append("text-negative")
    elif "pos" in sent:
        classes.append("text-positive")
    return " ".join(classes)


def _timeline_caption_text(selected_idx: int, total: int, *, virtual: bool) -> str:
    if virtual:
        return f"Valt segment {selected_idx + 1} av {total}"
    return f"Segment {selected_idx + 1} av {total}"


def render_timeline(
    enriched: list[dict[str, Any]],
    *,
    selected_idx: int,
    on_select: Callable[[int], None],
    virt_state: dict[str, Any],
) -> Label:
    """Render timeline; virtualized when segment count is large. Returns caption label."""
    total = len(enriched)
    if total == 0:
        ui.label("Inga segment.").classes("text-caption")
        return ui.label("")

    use_virtual = should_virtualize(total)
    if use_virtual:
        start, end = window_around_index(selected_idx, total)
        virt_state["timeline_start"] = start
        virt_state["timeline_end"] = end
    else:
        start, end = 0, total

    caption = ui.label(_timeline_caption_text(selected_idx, total, virtual=use_virtual)).classes(
        "text-caption q-mb-xs"
    )

    scroll = ui.scroll_area().classes("w-full timeline-panel")

    @ui.refreshable
    def timeline_body() -> None:
        s, e = (
            (start, end)
            if not use_virtual
            else (
                virt_state.get("timeline_start", 0),
                virt_state.get("timeline_end", total),
            )
        )
        if use_virtual and s > 0:
            ui.element("div").style(f"height: {s * TIMELINE_ROW_HEIGHT}px; width: 100%;")
        with ui.column().classes("w-full gap-0"):
            for i in range(s, min(e, total)):
                seg = enriched[i]
                is_selected = selected_idx == i
                label = (
                    f"[{i + 1}/{total}] {_segment_time_label(seg)} {seg.get('speaker', '?')}: "
                    f"{seg.get('text', '')[:60]}..."
                )

                def make_select(idx: int) -> Callable[[], None]:
                    def _select() -> None:
                        on_select(idx)
                        caption.set_text(_timeline_caption_text(idx, total, virtual=use_virtual))
                        if use_virtual:
                            ns, ne = window_around_index(idx, total)
                            virt_state["timeline_start"] = ns
                            virt_state["timeline_end"] = ne
                            timeline_body.refresh()

                    return _select

                ui.button(
                    label,
                    on_click=make_select(i),
                    color="primary" if is_selected else None,
                ).props("flat dense align=left").classes("w-full text-left")
        if use_virtual and e < total:
            ui.element("div").style(
                f"height: {max(0, total - e) * TIMELINE_ROW_HEIGHT}px; width: 100%;"
            )

    with scroll:
        timeline_body()

    return caption


def render_transcript_panel(
    enriched: list[dict[str, Any]],
    *,
    selected_idx: int,
    search: str,
    virt_state: dict[str, Any],
) -> None:
    """Render searchable transcript with virtual scroll for large calls."""
    indexed = filter_segments_with_index(enriched, search)
    total = len(indexed)
    if total == 0:
        ui.label("Inget transkript matchar sökningen." if search else "Inget transkript.").classes(
            "text-caption"
        )
        virt_state.pop("tx_refresh", None)
        return

    use_virtual = should_virtualize(total)
    row_height = TRANSCRIPT_ROW_HEIGHT

    if use_virtual:
        ui.label(f"Virtualiserat transkript · {total} segment (windowed rendering)").classes(
            "text-caption q-mb-xs"
        )

    virt_state["tx_indexed"] = indexed
    virt_state["tx_total"] = total
    if "tx_start" not in virt_state or virt_state.get("tx_search") != search:
        virt_state["tx_start"] = 0
        virt_state["tx_end"] = min(total, 25)
    virt_state["tx_search"] = search

    @ui.refreshable
    def transcript_body() -> None:
        items: list[tuple[int, dict[str, Any]]] = virt_state.get("tx_indexed", indexed)
        count = len(items)
        if not use_virtual:
            s, e = 0, count
        else:
            s = int(virt_state.get("tx_start", 0))
            e = int(virt_state.get("tx_end", min(count, 25)))

        if use_virtual and s > 0:
            ui.element("div").style(f"height: {s * row_height}px; width: 100%;")

        with ui.column().classes("w-full gap-1 q-pa-sm"):
            for i in range(s, min(e, count)):
                orig_idx, seg = items[i]
                text = seg.get("text", "")
                speaker = seg.get("speaker", "?")
                prefix = ">> " if selected_idx == orig_idx else ""
                css = _sentiment_css(seg)
                if search:
                    body = (
                        f"{prefix}<strong>{html.escape(speaker)}:</strong> "
                        f"{highlight_search_text(text, search)}"
                    )
                    ui.html(body).classes(css)
                else:
                    ui.markdown(f"{prefix}**{speaker}:** {text}").classes(css)

        if use_virtual and e < count:
            ui.element("div").style(f"height: {(count - e) * row_height}px; width: 100%;")

    def on_transcript_scroll(e: ScrollEventArguments) -> None:
        if not use_virtual:
            return
        new_start, new_end = compute_visible_range(
            e.vertical_position,
            e.vertical_container_size or 288,
            total,
            row_height,
        )
        if new_start != virt_state.get("tx_start") or new_end != virt_state.get("tx_end"):
            virt_state["tx_start"] = new_start
            virt_state["tx_end"] = new_end
            transcript_body.refresh()

    scroll = ui.scroll_area(on_scroll=on_transcript_scroll).classes("w-full transcript-panel")
    with scroll:
        transcript_body()

    virt_state["tx_scroll_area"] = scroll
    virt_state["tx_refresh"] = transcript_body.refresh


def scroll_transcript_to_index(virt_state: dict[str, Any], original_index: int) -> None:
    """Scroll virtual transcript so the segment at original_index is visible."""
    indexed: list[tuple[int, dict[str, Any]]] = virt_state.get("tx_indexed") or []
    scroll_area = virt_state.get("tx_scroll_area")
    refresh = virt_state.get("tx_refresh")
    if not indexed:
        return
    list_idx = next((i for i, (oid, _) in enumerate(indexed) if oid == original_index), None)
    if list_idx is None:
        return
    total = len(indexed)
    start, end = window_around_index(list_idx, total)
    virt_state["tx_start"] = start
    virt_state["tx_end"] = end
    if refresh:
        refresh()
    if scroll_area:
        scroll_area.scroll_to(pixels=scroll_pixels_for_index(start))
