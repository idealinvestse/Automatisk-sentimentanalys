"""Live pipeline analysis section (used by Testlabb).

Fas 3 – docs/archive/MIGRATION_TO_NICEGUI_PLAN.md §3
"""

from __future__ import annotations

import html
import json
from typing import Any

from nicegui import ui

from app.archive.nicegui_dashboard.services.nicegui_api_client import APIError
from app.archive.nicegui_dashboard.services.ui_helpers import (
    notify_api_error,
    notify_success,
    notify_warning,
)
from app.archive.nicegui_dashboard.state import DashboardState


def render_text_pipeline_section(state: DashboardState) -> None:
    """Render JSON segments → /analyze_pipeline test UI."""
    ui.label("Pipeline på JSON-segment").classes("text-subtitle1 q-mb-sm")
    client = state.api_client

    if client:
        ui.label(f"Backend: {client.base_url}").classes("text-caption")
        badge_color = "positive" if state.api_connected else "warning"
        ui.chip(
            "API ansluten" if state.api_connected else "API ej verifierad",
            color=badge_color,
        ).classes("q-mb-sm")
    else:
        ui.label("Ingen API-klient konfigurerad.").classes("text-caption text-warning")

    segments_input = ui.textarea(
        label="Klistra in segments (JSON)",
        placeholder='[{"text": "Hej, hur kan jag hjälpa dig?", "speaker": "Agent"}]',
    ).classes("w-full")

    use_llm = ui.checkbox("Använd LLM deep analysis", value=False)
    provider_dropdown = ui.select(
        label="LLM-provider",
        options=["openrouter", "groq"],
        value="openrouter",
    ).classes("w-48")
    provider_dropdown.bind_visibility_from(use_llm, "value")
    groq_gdpr_notice = ui.label(
        "⚠️ Groq: US/Saudi data centers (no EU hosting). Enable PII redaction."
    ).classes("text-caption text-warning")
    groq_gdpr_notice.set_visibility(False)

    def _on_provider_change(e: Any) -> None:
        groq_gdpr_notice.set_visibility(e.value == "groq" and bool(use_llm.value))

    provider_dropdown.on("update:model-value", _on_provider_change)
    use_llm.on(
        "update:model-value",
        lambda e: groq_gdpr_notice.set_visibility(
            provider_dropdown.value == "groq" and bool(e.value)
        ),
    )
    result_container = ui.column().classes("w-full q-mt-md")

    async def run_analysis() -> None:
        if not client:
            notify_api_error(Exception("API-klient saknas"))
            return

        raw = (segments_input.value or "").strip()
        if not raw:
            notify_warning("Ange segments som JSON")
            return

        try:
            segments: list[dict[str, Any]] = json.loads(raw)
            if not isinstance(segments, list) or not segments:
                raise ValueError("segments måste vara en icke-tom lista")
        except (json.JSONDecodeError, ValueError) as err:
            notify_api_error(err)
            return

        result_container.clear()
        with result_container:
            ui.spinner(size="lg")
            ui.label("Kör pipeline via backend...").classes("text-caption")

        try:
            if not state.api_connected:
                state.api_connected = await client.wait_for_health(attempts=3, interval=0.5)

            report = await client.analyze_pipeline(
                segments,
                use_mistral_llm=bool(use_llm.value),
                deep_analysis=bool(use_llm.value),
                provider=provider_dropdown.value,
            )
            state.api_connected = True

            result_container.clear()
            with result_container:
                ui.label("Analys klar").classes("text-subtitle1 text-positive")
                sent = report.get("sentiment_results") or []
                if sent:
                    labels = [str(s.get("label", "?")) for s in sent[:5]]
                    ui.label(f"Sentiment (första segment): {', '.join(labels)}").classes(
                        "text-body2"
                    )

                qa = (report.get("results") or {}).get("qa") or {}
                if qa.get("overall_qa_score") is not None:
                    ui.label(f"QA-poäng: {qa['overall_qa_score']}/100").classes("text-body2")

                llm = report.get("llm") or {}
                actionable = llm.get("actionable_summary") or {}
                if isinstance(actionable, dict) and actionable.get("problem"):
                    with ui.expansion("Actionable Summary", icon="insights"):
                        problem_text = html.escape(actionable.get("problem", ""))
                        ui.html(f"<strong>Problem:</strong> {problem_text}")

                with ui.expansion("Fullständigt svar (JSON)", icon="data_object"):
                    ui.code(json.dumps(report, indent=2, ensure_ascii=False, default=str))

            notify_success("Pipeline-analys slutförd")

        except APIError as err:
            result_container.clear()
            with result_container:
                ui.label("API-fel").classes("text-negative")
                ui.label(str(err)).classes("text-caption")
                if err.detail:
                    ui.code(str(err.detail)[:2000])
            notify_api_error(err)
        except Exception as err:
            result_container.clear()
            with result_container:
                ui.label(f"Fel: {err}").classes("text-negative")
            notify_api_error(err)

    ui.button("Analysera (pipeline)", color="primary", on_click=run_analysis).classes("q-mt-sm")


def render_live_analysis_tab(state: DashboardState) -> None:
    """Backward-compatible entry: full Live-analys tab (delegates to pipeline section)."""
    ui.label("Live-analys").classes("text-h6")
    render_text_pipeline_section(state)
