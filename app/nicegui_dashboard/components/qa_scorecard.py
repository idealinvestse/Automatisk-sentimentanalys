"""QA & Compliance scorecard section.

Fas 3 – docs/GROK_BUILD_PLAN_FAS1-3.md
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nicegui import ui

from app.nicegui_dashboard.components.call_detail import find_report
from app.nicegui_dashboard.services.fas4_data import format_evidence_spans, local_qa_from_report
from app.nicegui_dashboard.services.qa_display import qa_chip_color, qa_score_css_class
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
            with ui.card().classes("flex-1 min-w-[160px]"):
                ui.label("QA-poäng").classes("text-caption text-grey")
                ui.label(str(score if score is not None else "—")).classes(
                    f"text-h5 {qa_score_css_class(score)}"
                )
            with ui.card().classes("flex-1 min-w-[160px]"):
                ui.label("Status").classes("text-caption text-grey")
                ui.chip(
                    "Godkänd" if passed else "Underkänd",
                    color="positive" if passed else "negative",
                )
            with ui.card().classes("flex-1 min-w-[160px]"):
                ui.label("Risknivå").classes("text-caption text-grey")
                ui.chip(str(risk), color=qa_chip_color(score))

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
                        {"name": "evidence", "label": "Evidence", "field": "evidence"},
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

    qa_section()
    return qa_section.refresh