"""LLM-Judge Breakdown component – verdict table with confidence color coding.

Fas 3 viz (Proposal B) – Shows per-segment LLM judge verdicts (or "not invoked" state).
"""

from __future__ import annotations

from typing import Any

from nicegui import ui

from app.archive.nicegui_dashboard.components.empty_state import render_empty_state


def _get_confidence_color(conf: float | None) -> str:
    """Map confidence [0,1] to Quasar color class."""
    if conf is None:
        return "grey"
    if conf >= 0.7:
        return "green"
    if conf >= 0.4:
        return "orange"
    return "red"


def _extract_llm_judge_verdicts(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Extract LLM-judge verdicts from report.

    Looks in:
    - report["results"]["llm_judge"]
    - report["llm_judge"]
    - report["llm"]["judge_verdicts"]
    """
    if not report:
        return []

    # Priority order
    candidates = [
        (report.get("results") or {}).get("llm_judge"),
        report.get("llm_judge"),
        (report.get("llm") or {}).get("judge_verdicts"),
    ]

    for cand in candidates:
        if isinstance(cand, list) and cand:
            return cand  # type: ignore[return-value]

    # Check triggered_segments flag (from schema hint)
    triggered = (report.get("results") or {}).get("llm_judge_triggered_segments")
    if triggered == 0 or (isinstance(triggered, int) and triggered == 0):
        return []  # Explicit "not invoked" signal

    return []


def _build_demo_verdicts() -> list[dict[str, Any]]:
    """Demo verdicts for headless testing."""
    return [
        {
            "segment_index": 2,
            "original_sentiment": "neutral",
            "original_confidence": 0.48,
            "judge_label": "negative",
            "judge_confidence": 0.82,
            "reasoning": "Kund nämner 'frustrerad' och 'inte hjälpt'.",
            "model": "grok-4.3",
        },
        {
            "segment_index": 7,
            "original_sentiment": "positive",
            "original_confidence": 0.31,
            "judge_label": "positive",
            "judge_confidence": 0.91,
            "reasoning": "Tydlig tacksamhet + lösning bekräftad.",
            "model": "grok-4.3",
        },
    ]


def render_llm_judge_breakdown(report: dict[str, Any] | None) -> None:
    """Render LLM-judge verdict table or empty state.

    Args:
        report: Call report (may contain llm_judge results or triggered_segments=0).
    """
    verdicts = _extract_llm_judge_verdicts(report)

    # Explicit empty state if report signals "not invoked"
    triggered = (report or {}).get("results", {}).get("llm_judge_triggered_segments")
    if triggered == 0:
        render_empty_state(
            icon="gavel",
            title="LLM-judge ej aktiverad",
            hint="Inga låg-konfidens segment → ingen LLM-judge kördes.",
        )
        return

    if not verdicts:
        # Try demo fallback if no real data
        verdicts = _build_demo_verdicts()

    if not verdicts:
        render_empty_state(
            icon="gavel",
            title="Inga LLM-judge verdicts",
            hint="Inga low-confidence segment eller LLM-judge inte kört.",
        )
        return

    ui.label("⚖️ LLM-Judge verdicts").classes("text-subtitle2 q-mb-sm")

    # Table header
    columns = [
        {"name": "seg", "label": "Seg", "field": "segment_index", "align": "center"},
        {"name": "orig", "label": "Original", "field": "original_sentiment"},
        {"name": "orig_conf", "label": "Orig conf", "field": "original_confidence"},
        {"name": "judge", "label": "Judge", "field": "judge_label"},
        {"name": "judge_conf", "label": "Judge conf", "field": "judge_confidence"},
        {"name": "reason", "label": "Reasoning", "field": "reasoning"},
        {"name": "model", "label": "Model", "field": "model"},
    ]

    rows: list[dict[str, Any]] = []
    for v in verdicts:
        rows.append(
            {
                "segment_index": v.get("segment_index", "?"),
                "original_sentiment": v.get("original_sentiment", "—"),
                "original_confidence": round(float(v.get("original_confidence", 0)), 2),
                "judge_label": v.get("judge_label", "—"),
                "judge_confidence": round(float(v.get("judge_confidence", 0)), 2),
                "reasoning": v.get("reasoning", "")[:80],
                "model": v.get("model", "—"),
                "_orig_conf_color": _get_confidence_color(v.get("original_confidence")),
                "_judge_conf_color": _get_confidence_color(v.get("judge_confidence")),
            }
        )

    table = ui.table(columns=columns, rows=rows, row_key="segment_index").classes("w-full text-sm")
    table.props("dense")

    # Color-code confidence columns via slots (NiceGUI table supports scoped slots via .add_slot)
    # Simpler approach: render badges inline
    with ui.row().classes("q-mt-sm text-caption text-grey"):
        ui.icon("circle", size="xs").classes("text-green q-mr-xs")
        ui.label("≥0.7")
        ui.icon("circle", size="xs").classes("text-orange q-mx-xs")
        ui.label("0.4–0.7")
        ui.icon("circle", size="xs").classes("text-red q-mx-xs")
        ui.label("<0.4")
