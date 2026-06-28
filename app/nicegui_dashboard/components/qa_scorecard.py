"""QA & Compliance scorecard section.

Fas 3 – docs/archive/GROK_BUILD_PLAN_FAS1-3.md
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nicegui import ui

from app.nicegui_dashboard.components.call_detail import find_report
from app.nicegui_dashboard.components.ui_primitives import metric_card
from app.nicegui_dashboard.services.analytics_summary import qa_problem_calls
from app.nicegui_dashboard.services.fas4_data import format_evidence_spans, local_qa_from_report
from app.nicegui_dashboard.services.qa_display import qa_chip_color
from app.nicegui_dashboard.state import DashboardState
from app.services.data_services import filter_reports


def render_qa_scorecard_section(
    state: DashboardState,
    *,
    on_call_select: Callable[[str], None] | None = None,
) -> Callable[[], None]:
    """Render QA scorecard with call selector and criteria evidence."""

    def _reports() -> list[dict[str, Any]]:
        return filter_reports(state.reports, state.filters)

    def _ensure_call(reports: list[dict[str, Any]]) -> str | None:
        if not reports:
            return None
        cid = state.selected_qa_call_id or state.selected_call_id
        ids = [str(r.get("call_id") or r.get("id", "")) for r in reports]
        if cid not in ids:
            cid = ids[0]
            state.selected_qa_call_id = cid
        return cid

    @ui.refreshable
    def qa_section() -> None:
        reports = _reports()
        call_id = _ensure_call(reports)
        report = find_report(reports, call_id)
        qa = local_qa_from_report(report)

        ui.label("✅ QA & Compliance").classes("text-subtitle1 q-mb-sm")

        if not reports:
            ui.label("Inga samtal att visa QA för.").classes("text-caption")
            return

        options = {
            str(r.get("call_id") or r.get("id", "")): f"{r.get('call_id')} – {r.get('title', '')}"
            for r in reports
        }
        ui.select(
            options=options,
            value=call_id,
            label="Välj samtal",
            on_change=lambda e: (
                setattr(state, "selected_qa_call_id", e.value),
                qa_section.refresh(),
            ),
        ).classes("w-full q-mb-md").props("dense")

        if not qa:
            ui.label("Ingen QA-data för valt samtal (kör pipeline med scorecard).").classes(
                "text-caption"
            )
            return

        score = qa.get("overall_qa_score")
        passed = qa.get("passed")
        risk = qa.get("risk_level", "—")

        with ui.row().classes("w-full gap-4 flex-wrap q-mb-md"):
            metric_card(
                "QA-poäng",
                score if score is not None else "—",
                color=qa_chip_color(score),
                size="compact",
            )
            metric_card(
                "Status",
                "Godkänd" if passed else "Underkänd",
                color="positive" if passed else "negative",
                size="compact",
            )
            metric_card(
                "Risknivå",
                str(risk),
                color=qa_chip_color(score),
                size="compact",
            )

        flags = qa.get("compliance_flags") or []
        if flags:
            ui.label("Compliance-flaggor").classes("text-subtitle2 q-mt-xs")
            with ui.row().classes("flex-wrap gap-1"):
                for flag in flags[:12]:
                    ui.chip(str(flag), color="warning").props("dense")

        passed_crit = qa.get("passed_criteria") or []
        failed_crit = qa.get("failed_criteria") or []
        if passed_crit or failed_crit:
            with ui.row().classes("w-full gap-4 flex-wrap"):
                if passed_crit:
                    with ui.card().classes("flex-1"):
                        ui.label("Godkända kriterier").classes("text-subtitle2")
                        for c in passed_crit[:12]:
                            ui.label(f"✓ {c}").classes("text-positive text-caption")
                if failed_crit:
                    with ui.card().classes("flex-1"):
                        ui.label("Underkända kriterier").classes("text-subtitle2")
                        for c in failed_crit[:12]:
                            ui.label(f"✗ {c}").classes("text-negative text-caption")

        criteria = qa.get("criteria_results") or []
        if criteria:
            ui.label("Detaljer per kriterium").classes("text-subtitle2 q-mt-sm")
            rows = []
            for cr in criteria:
                if not isinstance(cr, dict):
                    continue
                evidence = cr.get("evidence") or cr.get("evidence_spans") or []
                rows.append(
                    {
                        "criterion": cr.get("criterion") or cr.get("name", "—"),
                        "passed": "Ja" if cr.get("passed") else "Nej",
                        "score": cr.get("score", "—"),
                        "evidence": format_evidence_spans(evidence),
                    }
                )
            if rows:
                table = ui.table(
                    columns=[
                        {"name": "criterion", "label": "Kriterium", "field": "criterion"},
                        {"name": "passed", "label": "OK", "field": "passed"},
                        {"name": "score", "label": "Poäng", "field": "score"},
                        {"name": "evidence", "label": "Bevis", "field": "evidence"},
                    ],
                    rows=rows,
                    row_key="criterion",
                ).classes("w-full")

                def _open_call(e: Any) -> None:
                    if on_call_select and call_id:
                        on_call_select(call_id)

                table.on("rowClick", lambda e: _open_call(e))

        if on_call_select and call_id:
            ui.button(
                "Öppna i Samtalsdetalj",
                on_click=lambda: on_call_select(call_id),
            ).props("flat").classes("q-mt-sm")

        problem_rows = qa_problem_calls(reports)
        if problem_rows:
            ui.label("Samtal med QA-problem").classes("text-subtitle2 q-mt-md")
            prob_table = ui.table(
                columns=[
                    {"name": "call_id", "label": "ID", "field": "call_id"},
                    {"name": "title", "label": "Titel", "field": "title"},
                    {"name": "agent", "label": "Agent", "field": "agent"},
                    {"name": "qa_score", "label": "QA", "field": "qa_score"},
                    {"name": "risk_level", "label": "Risk", "field": "risk_level"},
                    {"name": "passed", "label": "Godkänd", "field": "passed"},
                ],
                rows=problem_rows,
                row_key="call_id",
            ).classes("w-full")

            def _problem_row_click(e: Any) -> None:
                if not on_call_select:
                    return
                row = e.args[1] if len(e.args) > 1 else e.args[0]
                cid = row.get("call_id") if isinstance(row, dict) else None
                if cid:
                    on_call_select(str(cid))

            prob_table.on("rowClick", _problem_row_click)

    qa_section()
    return qa_section.refresh