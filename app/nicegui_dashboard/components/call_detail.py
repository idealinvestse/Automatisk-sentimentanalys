"""Call Detail View – header, timeline, transcript, structured insights.

Fas 6.1 – docs/MIGRATION_TO_NICEGUI_PLAN.md (virtualiserat transkript)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nicegui import ui

from app.nicegui_dashboard.components.virtual_transcript import (
    render_timeline,
    render_transcript_panel,
    scroll_transcript_to_index,
)
from app.nicegui_dashboard.state import DashboardState
from app.services.data_services import enrich_segments_with_sentiment, get_overall_sentiment


def find_report(reports: list[dict[str, Any]], call_id: str | None) -> dict[str, Any] | None:
    if not call_id:
        return None
    for report in reports:
        if report.get("call_id") == call_id:
            return report
    return None


def _format_duration(seconds: int | float | None) -> str:
    if not seconds:
        return "—"
    mins = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{mins} min {secs}s" if mins else f"{secs}s"


def _build_insights_markdown(report: dict[str, Any]) -> str:
    """Render LLM + Fas4 structured data as markdown."""
    parts: list[str] = []
    llm = report.get("llm") or {}
    results = report.get("results") or {}

    actionable = llm.get("actionable_summary") or {}
    if isinstance(actionable, dict) and actionable:
        problem = actionable.get("problem", "")
        if problem:
            parts.append(f"**Problem:** {problem}")
        recs = actionable.get("recommendations") or actionable.get("recommended_actions") or []
        if recs:
            parts.append("**Rekommendationer:**")
            for rec in recs[:5]:
                parts.append(f"- {rec}")

    assess = results.get("agent_assessment") or llm.get("agent_assessment") or {}
    if isinstance(assess, dict) and assess:
        empathy = assess.get("empathy_score")
        if empathy is not None:
            parts.append(f"**Empati-score:** {empathy}")
        strengths = assess.get("strengths") or assess.get("key_strengths") or []
        if strengths:
            parts.append(f"**Styrkor:** {', '.join(str(s) for s in strengths[:4])}")
        coaching = assess.get("specific_coaching_recommendations") or []
        if coaching:
            parts.append("**Coaching:**")
            for c in coaching[:3]:
                parts.append(f"- {c}")

    qa = results.get("qa") or results.get("compliance_qa") or {}
    if isinstance(qa, dict) and qa.get("overall_qa_score") is not None:
        parts.append(f"**QA-poäng:** {qa['overall_qa_score']}/100")

    alerts = results.get("alerts") or []
    if alerts:
        parts.append(f"**Alerts:** {len(alerts)} st")

    if not parts:
        summary = report.get("summary") or {}
        if isinstance(summary, dict) and summary.get("text"):
            parts.append(str(summary["text"]))
        else:
            parts.append("*Inga strukturerade insikter tillgängliga för detta samtal.*")

    return "\n\n".join(parts)


def render_call_detail_tab(
    state: DashboardState,
    *,
    on_back: Callable[[], None] | None = None,
) -> Callable[[], None]:
    """Render call detail; returns refresh callback for external updates."""

    detail_state: dict[str, Any] = {"selected_idx": 0, "search": ""}
    virt_state: dict[str, Any] = {}

    @ui.refreshable
    def _render_content() -> None:
        report = find_report(state.reports, state.selected_call_id)
        if not report:
            ui.label("🔍 Call Detail View").classes("text-h6")
            ui.label("Välj ett samtal i Översikt-tabellen.").classes("text-body2")
            return

        call_id = report.get("call_id", "UNKNOWN")
        meta = report.get("meta") or {}
        sentiment = get_overall_sentiment(report)
        qa = (report.get("results") or {}).get("qa") or {}
        qa_score = qa.get("overall_qa_score", "—")

        ui.label(f"🔍 Call Detail – {call_id}").classes("text-h6")
        with ui.card().classes("w-full"):
            ui.label(report.get("title", call_id)).classes("text-subtitle1")
            with ui.row().classes("gap-2 flex-wrap"):
                ui.chip(f"Agent: {meta.get('agent', 'Okänd')}", color="primary")
                ui.chip(f"Duration: {_format_duration(meta.get('duration_s'))}")
                ui.chip(f"Sentiment: {sentiment.get('label', 'neutral')}", color="secondary")
                ui.chip(f"QA: {qa_score}/100", color="warning")

        ui.separator()
        enriched = enrich_segments_with_sentiment(
            report.get("segments") or [],
            report.get("sentiment_results") or [],
        )
        n_seg = len(enriched)
        if n_seg >= 40:
            ui.chip(f"{n_seg} segment · virtualiserad vy", color="info").classes("q-mb-sm")

        def on_timeline_select(idx: int) -> None:
            detail_state["selected_idx"] = idx
            if virt_state.get("tx_refresh"):
                scroll_transcript_to_index(virt_state, idx)
            else:
                refresh_transcript.refresh()

        ui.label("Interaktiv Timeline").classes("text-subtitle2 q-mt-md")
        with ui.card().classes("w-full"):
            render_timeline(
                enriched,
                selected_idx=detail_state["selected_idx"],
                on_select=on_timeline_select,
                virt_state=virt_state,
            )

        ui.label("Transkript (sökbart)").classes("text-subtitle2 q-mt-md")

        @ui.refreshable
        def refresh_transcript() -> None:
            render_transcript_panel(
                enriched,
                selected_idx=detail_state["selected_idx"],
                search=detail_state.get("search", ""),
                virt_state=virt_state,
            )

        ui.input(
            "Sök i transkript",
            value=detail_state.get("search", ""),
            on_change=lambda e: (
                detail_state.update({"search": e.value or ""}),
                virt_state.pop("tx_start", None),
                virt_state.pop("tx_end", None),
                refresh_transcript.refresh(),
            ),
        ).classes("w-full")

        refresh_transcript()

        ui.label("Structured Insights (LLM + Fas4)").classes("text-subtitle2 q-mt-md")
        with ui.expansion("Actionable Summary & Agent Assessment", icon="insights").classes("w-full"):
            ui.markdown(_build_insights_markdown(report))

        with ui.row().classes("gap-2 q-mt-md"):
            ui.button("Lägg i coaching-kö", on_click=lambda: ui.notify("Coaching-kö (Fas 4)"))
            ui.button("Flagga samtal", on_click=lambda: ui.notify(f"Flaggat {call_id}"))
            if on_back:
                ui.button("Tillbaka till Översikt", on_click=on_back)

    _render_content()

    def refresh() -> None:
        _render_content.refresh()

    return refresh