"""Call Detail View – header, timeline, transcript, structured insights."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from nicegui import ui

from app.nicegui_dashboard.components.call_alerts_section import render_call_alerts_section
from app.nicegui_dashboard.components.emotion_timeline import render_emotion_timeline
from app.nicegui_dashboard.components.empty_state import render_empty_state
from app.nicegui_dashboard.components.evidence_panel import render_evidence_panel
from app.nicegui_dashboard.components.llm_judge_panel import render_llm_judge_panel
from app.nicegui_dashboard.components.pii_audit import render_pii_audit_panel
from app.nicegui_dashboard.components.ui_primitives import render_section_title
from app.nicegui_dashboard.components.virtual_transcript import (
    render_timeline,
    render_transcript_panel,
    scroll_transcript_to_index,
)
from app.nicegui_dashboard.services.analytics_summary import compute_call_snapshot, summarize_emotions
from app.nicegui_dashboard.services.chart_data import (
    build_trajectory_figure,
    segment_index_from_trajectory_x,
)
from app.nicegui_dashboard.services.qa_display import qa_chip_color
from app.nicegui_dashboard.services.transcript_virtualizer import filter_segments_with_index
from app.nicegui_dashboard.state import DashboardState
from app.services.data_services import enrich_segments_with_sentiment, get_overall_sentiment


def find_report(reports: list[dict[str, Any]], call_id: str | None) -> dict[str, Any] | None:
    if not call_id:
        return None
    needle = str(call_id)
    for report in reports:
        rid = str(report.get("call_id") or report.get("id") or "")
        if rid == needle:
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

    root = llm.get("root_cause") or {}
    root_text = (
        root.get("primary_cause") or root.get("summary") if isinstance(root, dict) else None
    )
    if root_text:
        parts.append(f"**Rotorsak:** {root_text}")
    elif not llm:
        local_root = results.get("root_cause") or {}
        if isinstance(local_root, dict) and local_root.get("top_root_cause"):
            parts.append(f"**Rotorsak (lokal):** {local_root['top_root_cause']}")

    traj = llm.get("trajectory") or {}
    if isinstance(traj, dict) and traj.get("summary"):
        parts.append(f"**Kundresa:** {traj['summary']}")

    actionable = llm.get("actionable_summary") or {}
    if isinstance(actionable, dict) and actionable:
        problem = actionable.get("problem", "")
        if problem:
            parts.append(f"**Problem:** {problem}")
        recs = (
            actionable.get("recommendations_for_qa")
            or actionable.get("recommendations")
            or actionable.get("recommended_actions")
            or []
        )
        if recs:
            parts.append("**Rekommendationer:**")
            for rec in recs[:5]:
                parts.append(f"- {rec}")

    if not llm:
        coaching = results.get("actionable_coaching") or {}
        if isinstance(coaching, dict) and coaching.get("top_recommendation"):
            parts.append(f"**Coaching (lokal):** {coaching['top_recommendation']}")

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
        parts.append(f"**Aviseringar:** {len(alerts)} st")

    if not parts:
        summary = report.get("summary") or {}
        if isinstance(summary, dict) and summary.get("text"):
            parts.append(str(summary["text"]))
        else:
            parts.append("*Inga strukturerade insikter tillgängliga för detta samtal.*")

    return "\n\n".join(parts)


def _render_intent_topics(report: dict[str, Any]) -> None:
    intents = report.get("intent_results") or []
    topics = (report.get("results") or {}).get("topics") or report.get("topics") or []
    if isinstance(topics, dict):
        topics = topics.get("topics") or []
    if not intents and not topics:
        return
    with ui.expansion("Intent & ämnen", icon="label", value=False).classes("w-full"):
        if intents:
            ui.label("Intent").classes("text-caption text-grey")
            with ui.row().classes("flex-wrap gap-1 q-mb-sm"):
                for item in intents[:12]:
                    if isinstance(item, dict):
                        label = item.get("intent") or item.get("label") or "?"
                        ui.chip(str(label), color="info").classes("text-caption")
        if topics:
            ui.label("Ämnen").classes("text-caption text-grey")
            with ui.row().classes("flex-wrap gap-1"):
                for t in topics[:12]:
                    if isinstance(t, dict):
                        word = t.get("word") or t.get("topic") or "?"
                        ui.chip(str(word), color="primary").classes("text-caption")


_BACK_LABELS = {
    "overview": "Tillbaka till Översikt",
    "analytics": "Tillbaka till Analys & Trender",
    "agent_performance": "Tillbaka till Agentprestanda",
    "fas4": "Tillbaka till Fas 4 Insikter",
}


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
            render_empty_state(
                icon="call",
                title="Inget samtal valt",
                hint="Välj ett samtal i tabellen under Översikt.",
            )
            return

        call_id = report.get("call_id", "UNKNOWN")
        meta = report.get("meta") or {}
        sentiment = get_overall_sentiment(report)
        qa = (report.get("results") or {}).get("qa") or {}
        qa_score = qa.get("overall_qa_score", "—")
        snapshot = compute_call_snapshot(report)

        with ui.row().classes("w-full items-center justify-between"):
            ui.label(f"Samtalsdetalj – {call_id}").classes("text-h6")
            ui.button(
                "Ladda ner JSON",
                icon="download",
                on_click=lambda: ui.download(
                    json.dumps(report, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
                    f"{call_id}_rapport.json",
                ),
            ).props("outline dense")

        with ui.card().classes("w-full"):
            ui.label(report.get("title", call_id)).classes("text-subtitle1")
            with ui.row().classes("gap-2 flex-wrap"):
                ui.chip(f"Agent: {meta.get('agent', 'Okänd')}", color="primary")
                ui.chip(f"Längd: {_format_duration(meta.get('duration_s'))}")
                ui.chip(f"Sentiment: {sentiment.get('label', 'neutral')}", color="secondary")
                ui.chip(f"QA: {qa_score}/100", color=qa_chip_color(qa_score))

        if snapshot:
            with ui.card().classes("w-full q-mt-sm"):
                render_section_title("Samtalsöversikt", icon="insights")
                with ui.row().classes("gap-2 flex-wrap"):
                    ui.chip(f"Kategori: {snapshot.get('category', '—')}")
                    ui.chip(f"Trend: {snapshot.get('trajectory_trend', '—')}", color="info")
                    ui.chip(f"Aviseringar: {snapshot.get('alert_count', 0)}")
                    ui.chip(f"Segment: {snapshot.get('segment_count', 0)}")
                    if snapshot.get("trajectory_min") is not None:
                        ui.chip(
                            f"Sentiment: {snapshot['trajectory_min']} … {snapshot['trajectory_max']}"
                        )
                    ui.chip(f"Negativa toppar: {snapshot.get('negative_peaks', 0)}", color="negative")
                emotions = summarize_emotions(report)
                if emotions:
                    with ui.row().classes("gap-1 flex-wrap q-mt-xs"):
                        for emo in emotions[:3]:
                            ui.chip(f"{emo['label_sv']} {emo['avg']:.2f}", color="accent").classes(
                                "text-caption"
                            )

        ui.separator()
        results = report.get("results") or {}
        enriched = enrich_segments_with_sentiment(
            report.get("segments") or [],
            report.get("sentiment_results") or [],
            emotion_results=results.get("emotion") or report.get("emotion_results"),
            compliance_risk=results.get("compliance_risk"),
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

        render_section_title("Kundsentiment över tid", icon="show_chart")
        traj_plot = ui.plotly(build_trajectory_figure(report)).classes("w-full chart-container")

        def _on_traj_click(e: Any) -> None:
            args = getattr(e, "args", e) or {}
            points = args.get("points") or []
            if not points:
                return
            pt = points[0] if isinstance(points, list) else points
            x_val = pt.get("x") if isinstance(pt, dict) else None
            if x_val is not None:
                idx = segment_index_from_trajectory_x(report, float(x_val))
                on_timeline_select(idx)

        traj_plot.on("plotly_click", _on_traj_click)

        render_section_title("Interaktiv tidslinje", icon="timeline")
        with ui.card().classes("w-full"):
            render_timeline(
                enriched,
                selected_idx=detail_state["selected_idx"],
                on_select=on_timeline_select,
                virt_state=virt_state,
            )

        render_section_title("Transkript (sökbart)", icon="article")

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
                _update_search_hint(),
                refresh_transcript.refresh(),
            ),
        ).classes("w-full")
        search_hint = ui.label("").classes("text-caption q-mb-xs")

        def _update_search_hint() -> None:
            query = detail_state.get("search", "").strip()
            if not query:
                search_hint.set_text("")
                return
            hits = len(filter_segments_with_index(enriched, query))
            search_hint.set_text(f"{hits} segment matchar «{query}»")

        _update_search_hint()
        refresh_transcript()

        render_section_title("Strukturerade insikter", icon="psychology")
        with ui.expansion("Sammanfattning & agentbedömning", icon="insights").classes("w-full"):
            ui.markdown(_build_insights_markdown(report))

        render_evidence_panel(report)
        render_call_alerts_section(report)
        render_pii_audit_panel(report.get("results"))
        _render_intent_topics(report)

        with ui.expansion("Känsloprofil", icon="mood", value=False).classes("w-full"):
            render_emotion_timeline(report)

        render_llm_judge_panel((report.get("results") or {}).get("llm_judge"))

        if on_back:
            back_label = _BACK_LABELS.get(
                state.detail_source_tab,
                _BACK_LABELS["overview"],
            )
            with ui.row().classes("gap-2 q-mt-md"):
                ui.button(back_label, on_click=on_back)

    _render_content()

    def refresh() -> None:
        _render_content.refresh()

    return refresh