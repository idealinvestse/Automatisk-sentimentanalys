"""LLM Judge Panel component for Fas 4 Dashboard.

Shows LLM re-evaluation of low-confidence sentiment segments.
Polished card-based UI with filter for changed verdicts.
"""

from __future__ import annotations

from typing import Any

from nicegui import ui

from app.nicegui_dashboard.components.empty_state import render_empty_state


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
    return orig != judge and judge not in ("", "?")


def render_llm_judge_panel(
    llm_judge_result: dict[str, Any] | None,
    title: str = "LLM Judge – Lågkonfidens segment",
) -> None:
    """Render polished LLM Judge verdicts panel with filter."""
    verdicts = _get_verdicts(llm_judge_result)

    with ui.card().classes("w-full"):
        ui.label(title).classes("text-lg font-semibold mb-1")

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
        with ui.row().classes("items-center justify-between w-full mb-2"):
            ui.markdown(f"**{total}** segment bedömdes | **{changed_count}** ändrade").classes("text-sm text-gray-600")

            filter_mode = ui.toggle(
                ["Alla", "Endast ändrade"],
                value="Alla",
                on_change=lambda e: refresh_panel.refresh(),
            ).props("dense toggle-color=primary").classes("text-sm")

        @ui.refreshable
        def refresh_panel() -> None:
            show_only_changed = filter_mode.value == "Endast ändrade"
            filtered = [v for v in verdicts if not show_only_changed or _is_changed(v)]

            if not filtered:
                ui.label("Inga ändrade bedömningar i detta urval.").classes("text-caption text-grey q-my-md")
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

                with ui.card().classes("mb-2 p-2 q-pa-sm"):
                    with ui.row().classes("items-center justify-between"):
                        ui.badge(f"Seg {seg_idx}", color="dark").classes("text-xs")
                        ui.html(f"<b>Original:</b> {orig_label} <span class='text-caption'>({orig_conf:.2f})</span>")
                        ui.badge(str(judge_label).upper(), color=color).classes("text-sm")

                    ui.html(
                        f"<b>LLM:</b> {judge_label} <span style='color:{color}'>({judge_conf:.2f})</span>"
                    ).classes("text-sm q-mt-xs")

                    if reasoning:
                        with ui.expansion("Motivering", icon="info").classes("w-full text-sm q-mt-xs"):
                            ui.markdown(reasoning)

        refresh_panel()


def render_llm_judge_summary(llm_judge_result: dict[str, Any] | None) -> None:
    """Compact summary for headers."""
    verdicts = _get_verdicts(llm_judge_result)
    if not verdicts:
        return
    changed = sum(1 for v in verdicts if _is_changed(v))
    ui.badge(f"LLM Judge: {len(verdicts)} ({changed} ändrade)", color="blue").classes("text-xs")