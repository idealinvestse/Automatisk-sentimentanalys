"""LLM Judge Panel component for Fas 4 Dashboard.

Shows LLM re-evaluation of low-confidence sentiment segments.
Polished card-based UI with filter for changed verdicts.
"""

from __future__ import annotations

import html
from typing import Any

from nicegui import ui

from app.archive.nicegui_dashboard.components.empty_state import render_empty_state


def _get_verdicts(llm_judge_result: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Extract verdicts list from llm_judge result."""
    if not llm_judge_result:
        return []
    if isinstance(llm_judge_result, list):
        return llm_judge_result
    if isinstance(llm_judge_result, dict):
        return llm_judge_result.get("verdicts", []) or llm_judge_result.get("results", [])
    return []


def _is_changed(v: dict[str, Any]) -> bool:
    """Check if judge changed the original label."""
    orig = str(v.get("original_sentiment", v.get("original_label", ""))).lower()
    judge = str(v.get("judge_label", v.get("label", ""))).lower()
    if not orig or judge in ("", "?"):
        return False
    return orig != judge


def render_llm_judge_panel(
    llm_judge_result: dict[str, Any] | None,
    title: str = "LLM Judge – Lågkonfidens segment",
) -> None:
    """Render polished LLM Judge verdicts panel with filter."""
    verdicts = _get_verdicts(llm_judge_result)

    with ui.card().classes("w-full"):
        ui.label(title).classes("text-subtitle1 q-mb-sm")

        if not verdicts:
            render_empty_state(
                icon="psychology",
                title="Inga LLM-judge bedömningar",
                hint="Inga lågkonfidens segment eller LLM-judge inte kört ännu.",
            )
            return

        changed_count = sum(1 for v in verdicts if _is_changed(v))
        total = len(verdicts)

        # Header with stats + filter
        with ui.row().classes("items-center justify-between w-full q-mb-sm"):
            ui.markdown(f"**{total}** segment bedömdes | **{changed_count}** ändrade").classes(
                "text-caption text-grey"
            )

            with ui.row().classes("items-center gap-1"):
                filter_mode = (
                    ui.toggle(
                        ["Alla", "Endast ändrade"],
                        value="Alla",
                        on_change=lambda e: refresh_panel.refresh(),
                    )
                    .props("dense toggle-color=primary")
                    .classes("text-caption")
                )

                ui.icon("help_outline", size="xs").classes("text-grey cursor-help").tooltip(
                    "Visar endast segment där LLM:en ändrade den ursprungliga sentiment-bedömningen (t.ex. neutral → negative)."
                )

        @ui.refreshable
        def refresh_panel() -> None:
            show_only_changed = filter_mode.value == "Endast ändrade"
            filtered = [v for v in verdicts if not show_only_changed or _is_changed(v)]

            if not filtered:
                ui.label("Inga ändrade bedömningar i detta urval.").classes(
                    "text-caption text-grey q-my-md"
                )
                return

            for v in filtered:
                orig_label = v.get("original_sentiment", v.get("original_label", "?"))
                orig_conf = float(v.get("original_confidence", v.get("original_score", 0.0)))
                judge_label = v.get("judge_label", v.get("label", "?"))
                judge_conf = float(v.get("judge_confidence", v.get("score", 0.0)))
                reasoning = v.get("reasoning", v.get("explanation", ""))
                seg_idx = v.get("segment_index", v.get("idx", "?"))

                # Color
                j_lower = str(judge_label).lower()
                if j_lower in ("positive", "positiv"):
                    color = "green"
                elif j_lower in ("negative", "negativ"):
                    color = "red"
                else:
                    color = "grey"

                with ui.card().classes("q-mb-sm q-pa-sm"):
                    with ui.row().classes("items-center justify-between"):
                        ui.badge(f"Seg {seg_idx}", color="dark").classes("text-caption")
                        ui.html(
                            f"<b>Original:</b> {orig_label} <span class='text-caption'>({orig_conf:.2f})</span>"
                        )
                        ui.badge(str(judge_label).upper(), color=color).classes("text-caption")

                    with ui.row().classes("items-center gap-2 q-mt-xs"):
                        ui.label("LLM:").classes("text-caption")
                        ui.badge(str(judge_label).upper(), color=color).classes("text-caption")
                        ui.badge(f"{judge_conf:.2f}", color=color).classes("text-caption")

                    if reasoning:
                        with ui.expansion("Motivering", icon="info").classes(
                            "w-full text-caption q-mt-xs"
                        ):
                            ui.html(html.escape(reasoning))

        refresh_panel()


def render_llm_judge_summary(llm_judge_result: dict[str, Any] | None) -> None:
    """Compact summary for headers."""
    verdicts = _get_verdicts(llm_judge_result)
    if not verdicts:
        return
    changed = sum(1 for v in verdicts if _is_changed(v))
    ui.badge(f"LLM Judge: {len(verdicts)} ({changed} ändrade)", color="blue").classes("text-xs")
