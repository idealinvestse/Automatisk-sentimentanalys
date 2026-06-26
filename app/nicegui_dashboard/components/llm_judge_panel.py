"""LLM Judge Panel component for Fas 4 Dashboard.

Shows LLM re-evaluation of low-confidence sentiment segments.
Clean, Swedish, and follows existing component patterns.
"""

from __future__ import annotations

from typing import Any

from nicegui import ui

from app.nicegui_dashboard.components.empty_state import render_empty_state


def _get_verdicts(llm_judge_result: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Extract verdicts list from llm_judge result (handles different shapes)."""
    if not llm_judge_result:
        return []

    # Common shapes: {"verdicts": [...] } or directly a list
    if isinstance(llm_judge_result, list):
        return llm_judge_result
    if isinstance(llm_judge_result, dict):
        return llm_judge_result.get("verdicts", []) or llm_judge_result.get("results", [])
    return []


def render_llm_judge_panel(
    llm_judge_result: dict[str, Any] | None,
    title: str = "LLM Judge – Lågkonfidens segment",
) -> None:
    """Render LLM Judge verdicts panel.

    Args:
        llm_judge_result: Result from pipeline (results["llm_judge"]) or API.
        title: Card title.
    """
    verdicts = _get_verdicts(llm_judge_result)

    with ui.card().classes("w-full"):
        ui.label(title).classes("text-lg font-semibold mb-2")

        if not verdicts:
            render_empty_state(
                icon="psychology",
                title="Inga LLM-judge bedömningar",
                hint="Välj ett samtal med lågkonfidens sentiment-segment eller vänta på pipeline-wiring.",
            )
            return

        # Summary row
        triggered = len(verdicts)
        ui.markdown(f"**{triggered}** segment bedömdes om av LLM:en").classes("mb-3 text-sm text-gray-600")

        # Verdicts table / cards
        for v in verdicts:
            original_label = v.get("original_sentiment", v.get("original_label", "?"))
            original_conf = v.get("original_confidence", v.get("original_score", 0.0))
            judge_label = v.get("judge_label", v.get("label", "?"))
            judge_conf = v.get("judge_confidence", v.get("score", 0.0))
            reasoning = v.get("reasoning", v.get("explanation", "Ingen motivering."))
            segment_text = v.get("text", v.get("segment_text", ""))[:180]

            # Color coding
            if judge_label.lower() in ("positive", "positiv"):
                color = "green"
            elif judge_label.lower() in ("negative", "negativ"):
                color = "red"
            else:
                color = "gray"

            with ui.card().classes("mb-2 p-3"):
                # Header: original vs judge
                with ui.row().classes("items-center justify-between w-full"):
                    ui.html(
                        f"<b>Original:</b> {original_label} "
                        f"(<span class='text-gray-500'>{original_conf:.2f}</span>)"
                    )
                    ui.badge(judge_label.upper(), color=color).classes("text-sm")

                # Judge confidence
                ui.html(
                    f"<b>LLM-bedömning:</b> {judge_label} "
                    f"(<span style='color: {color}'>{judge_conf:.2f}</span>)"
                ).classes("text-sm mb-1")

                # Reasoning
                if reasoning:
                    with ui.expansion("Motivering", icon="info").classes("w-full text-sm"):
                        ui.markdown(reasoning)

                # Optional: show segment text if available
                if segment_text:
                    ui.html(f"<i>Segment:</i> <span class='text-gray-600'>{segment_text}...</span>").classes("text-xs mt-1")


def render_llm_judge_summary(llm_judge_result: dict[str, Any] | None) -> None:
    """Compact summary badge for header or overview cards."""
    verdicts = _get_verdicts(llm_judge_result)
    if not verdicts:
        return

    count = len(verdicts)
    ui.badge(f"LLM Judge: {count} segment", color="blue").classes("text-xs")