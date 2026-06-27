"""Testlabb tab – audio samples, text/pipeline tests, system checks, reports."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Callable

from nicegui import ui

from app.nicegui_dashboard.components.live_analysis import render_text_pipeline_section
from app.nicegui_dashboard.components.ui_primitives import render_section_title, render_tab_header
from app.nicegui_dashboard.services.nicegui_api_client import APIError
from app.nicegui_dashboard.services.test_lab_service import (
    default_run_settings,
    list_audio_reports,
    list_emotion_options,
    list_pack_options,
    list_sample_rows,
    load_examples_txt,
    load_report_json,
    repo_root,
    resolve_api_audio_path,
    run_doctor_check,
    run_scenario_ui,
    run_single_asr,
    run_single_pipeline,
    run_single_sentiment_chain,
    validate_catalog,
)
from app.nicegui_dashboard.services.ui_helpers import (
    notify_api_error,
    notify_error,
    notify_success,
    notify_warning,
)
from app.nicegui_dashboard.state import DashboardState

_SCENARIO_OPTIONS = [
    {"label": "Smoke (3 filer)", "value": "smoke"},
    {"label": "ASR (8 känslor)", "value": "asr"},
    {"label": "Pipeline (2 filer)", "value": "pipeline"},
    {"label": "Sentimentkedja", "value": "sentiment_chain"},
    {"label": "Språk-sanity", "value": "language_sanity"},
    {"label": "Katalog (ingen ML)", "value": "catalog"},
]


def render_test_lab_tab(state: DashboardState) -> None:
    """Render consolidated test lab with nested sub-tabs."""
    render_tab_header(
        "Testlabb",
        hint="Testa ljudprover från samples/audio, pipeline, systemhälsa och sparade rapporter.",
    )

    results_holder: dict[str, Any] = {"data": None, "container": None}

    with ui.tabs().classes("w-full") as sub_tabs:
        audio_tab = ui.tab("Ljudprover", icon="audiotrack")
        text_tab = ui.tab("Text & pipeline", icon="text_fields")
        system_tab = ui.tab("System", icon="health_and_safety")
        reports_tab = ui.tab("Rapporter", icon="description")

    with ui.tab_panels(sub_tabs, value=audio_tab).classes("w-full"):
        with ui.tab_panel(audio_tab):
            _render_audio_section(state, results_holder)
        with ui.tab_panel(text_tab):
            _render_text_section(state)
        with ui.tab_panel(system_tab):
            _render_system_section(state)
        with ui.tab_panel(reports_tab):
            _render_reports_section(results_holder)


def _render_results_panel(container: ui.column, data: Any, *, title: str = "Resultat") -> None:
    container.clear()
    with container:
        ui.label(title).classes("text-subtitle1 q-mb-sm")
        if data is None:
            ui.label("Inga resultat ännu.").classes("text-caption")
            return
        if isinstance(data, dict) and data.get("files"):
            summary = data.get("summary") or {}
            ui.label(
                f"Scenario: {data.get('scenario', '?')} · "
                f"{data.get('n_files', 0)} filer · "
                f"{data.get('duration_s', 0)}s"
            ).classes("text-caption")
            if summary.get("sentiment_accuracy") is not None:
                ui.label(
                    f"Sentiment-träff: {summary['sentiment_accuracy']:.0%} "
                    f"({summary.get('sentiment_compared', 0)} jämförda)"
                ).classes("text-body2")
            rows = []
            for f in data["files"]:
                rows.append(
                    {
                        "path": f.get("relative_path", "?"),
                        "ok": "✓" if f.get("ok") else "✗",
                        "expected": f.get("expected_sentiment") or "—",
                        "pred": f.get("sentiment_pred") or "—",
                        "preview": (f.get("transcript_preview") or f.get("error") or "—")[:80],
                    }
                )
            ui.table(
                columns=[
                    {"name": "path", "label": "Fil", "field": "path", "align": "left"},
                    {"name": "ok", "label": "OK", "field": "ok"},
                    {"name": "expected", "label": "Förväntat", "field": "expected"},
                    {"name": "pred", "label": "Faktiskt", "field": "pred"},
                    {"name": "preview", "label": "Förhandsvisning", "field": "preview", "align": "left"},
                ],
                rows=rows,
                row_key="path",
            ).classes("w-full")
            with ui.expansion("Fullständig JSON", icon="data_object"):
                ui.code(json.dumps(data, indent=2, ensure_ascii=False, default=str))
        elif isinstance(data, list):
            ui.table(
                columns=[
                    {"name": "path", "label": "Fil", "field": "path", "align": "left"},
                    {"name": "ok", "label": "OK", "field": "ok"},
                    {"name": "detail", "label": "Detalj", "field": "detail", "align": "left"},
                ],
                rows=data,
                row_key="path",
            ).classes("w-full")
        elif isinstance(data, dict):
            for key, value in data.items():
                if key == "transcript" and isinstance(value, str) and len(value) > 200:
                    ui.label(f"{key}: {value[:200]}…").classes("text-body2")
                else:
                    ui.label(f"{key}: {value}").classes("text-body2")
            with ui.expansion("JSON", icon="data_object"):
                ui.code(json.dumps(data, indent=2, ensure_ascii=False, default=str))
        else:
            ui.label(str(data)).classes("text-body2")


def _render_audio_section(state: DashboardState, results_holder: dict[str, Any]) -> None:
    defaults = default_run_settings()
    pack_options = list_pack_options()
    emotion_options = list_emotion_options()

    validation = validate_catalog()
    status_color = "positive" if validation.ok else "negative"
    ui.chip(
        f"Katalog: {'OK' if validation.ok else 'Fel'}",
        color=status_color,
    ).classes("q-mb-sm")
    if not validation.ok and validation.errors:
        ui.label(validation.errors[0]).classes("text-caption text-negative")

    filter_state: dict[str, Any] = {
        "pack": pack_options[0]["value"] if pack_options else None,
        "emotion": None,
        "search": "",
        "limit": 50,
    }

    pack_opts = {p["value"]: p["label"] for p in pack_options}
    emotion_opts: dict[str, str] = {"": "Alla"}
    emotion_opts.update({e: e for e in emotion_options})

    with ui.row().classes("w-full gap-4 q-mb-md flex-wrap"):
        pack_select = ui.select(
            label="Paket",
            options=pack_opts,
            value=filter_state["pack"],
        ).classes("min-w-card")
        emotion_select = ui.select(
            label="Känsla",
            options=emotion_opts,
            value="",
        ).classes("min-w-card")
        search_input = ui.input(label="Sök sökväg", placeholder="Actor_01").classes("min-w-48")
        limit_input = ui.number(label="Max rader", value=50, min=5, max=500).classes("w-28")

    run_mode = ui.toggle(["Lokal", "API"], value="Lokal").classes("q-mb-sm")

    with ui.expansion("ASR-inställningar", icon="tune").classes("w-full q-mb-md"):
        with ui.row().classes("gap-4 flex-wrap"):
            backend_input = ui.select(
                label="Motor",
                options=["faster", "transformers", "whisperx"],
                value=defaults["backend"],
            ).classes("w-40")
            model_input = ui.input(label="Modell", value=defaults["model"]).classes("w-48")
            language_input = ui.input(label="Språk", value="en").classes("w-24")
            device_input = ui.select(
                label="Enhet",
                options=["auto", "cpu", "cuda", "cuda:0", "mps"],
                value=defaults["device"],
            ).classes("w-32")

    table_container = ui.column().classes("w-full")
    results_container = ui.column().classes("w-full q-mt-md")
    results_holder["container"] = results_container

    selected_rows: list[dict[str, Any]] = []

    def _settings() -> dict[str, str]:
        return {
            "backend": str(backend_input.value or "faster"),
            "model": str(model_input.value or defaults["model"]),
            "device": str(device_input.value or "cpu"),
            "language": str(language_input.value or "en"),
        }

    def _refresh_table() -> None:
        nonlocal selected_rows
        emotion_val = emotion_select.value or None
        rows = list_sample_rows(
            pack_id=pack_select.value,
            emotions=[emotion_val] if emotion_val else None,
            search=search_input.value,
            limit=int(limit_input.value or 50),
        )
        selected_rows = []
        table_container.clear()
        with table_container:
            ui.label(f"{len(rows)} provfiler").classes("text-caption q-mb-xs")
            table = ui.table(
                columns=[
                    {"name": "path", "label": "Sökväg", "field": "path", "align": "left"},
                    {"name": "emotion", "label": "Känsla", "field": "emotion"},
                    {"name": "expected", "label": "Förväntat", "field": "expected"},
                    {"name": "statement", "label": "Text", "field": "statement", "align": "left"},
                    {"name": "actor", "label": "Talare", "field": "actor"},
                ],
                rows=rows,
                row_key="id",
                selection="multiple",
            ).classes("w-full")
            table.on("selection", lambda e: _on_selection(e, rows))

    def _on_selection(event: dict, rows: list[dict[str, Any]]) -> None:
        nonlocal selected_rows
        keys = set(event.get("rows") or [])
        selected_rows = [r for r in rows if r["id"] in keys]

    def _targets() -> list[dict[str, Any]]:
        if selected_rows:
            return selected_rows
        rows = list_sample_rows(
            pack_id=pack_select.value,
            emotions=[emotion_select.value] if (emotion_select.value or "") else None,
            search=search_input.value,
            limit=1,
        )
        return rows[:1]

    async def _run_local(action: str) -> None:
        targets = _targets()
        if not targets:
            notify_warning("Inga filer valda")
            return
        settings = _settings()
        results_container.clear()
        with results_container:
            ui.spinner(size="lg")
            ui.label(f"Kör {action} lokalt...").classes("text-caption")

        out_rows: list[dict[str, Any]] = []
        for row in targets:
            try:
                if action == "asr":
                    res = await asyncio.to_thread(
                        run_single_asr,
                        row["abs_path"],
                        backend=settings["backend"],
                        device=settings["device"],
                        language=settings["language"],
                    )
                    detail = (res.get("transcript") or "")[:120]
                    ok = res.get("ok")
                elif action == "pipeline":
                    res = await asyncio.to_thread(
                        run_single_pipeline,
                        row["abs_path"],
                        backend=settings["backend"],
                        device=settings["device"],
                        language=settings["language"],
                    )
                    detail = f"sentiment={res.get('sentiment')}"
                    ok = res.get("ok")
                else:
                    res = await asyncio.to_thread(
                        run_single_sentiment_chain,
                        row["abs_path"],
                        backend=settings["backend"],
                        device=settings["device"],
                        language=settings["language"],
                        expected_sentiment=row.get("expected") if row.get("expected") != "—" else None,
                    )
                    detail = f"pred={res.get('sentiment_pred')} (förv: {row.get('expected')})"
                    ok = res.get("ok")
                out_rows.append({"path": row["path"], "ok": "✓" if ok else "✗", "detail": detail})
            except Exception as exc:
                out_rows.append({"path": row["path"], "ok": "✗", "detail": str(exc)})

        results_holder["data"] = out_rows
        _render_results_panel(results_container, out_rows)
        notify_success(f"{action} klar ({len(out_rows)} filer)")

    async def _run_api(action: str) -> None:
        client = state.api_client
        if not client:
            notify_error("API-klient saknas")
            return
        targets = _targets()
        if not targets:
            notify_warning("Inga filer valda")
            return
        settings = _settings()
        media_root = os.environ.get("API_MEDIA_ROOT")

        results_container.clear()
        with results_container:
            ui.spinner(size="lg")
            ui.label(f"Kör {action} via API...").classes("text-caption")

        try:
            if not state.api_connected:
                state.api_connected = await client.wait_for_health(attempts=3, interval=0.5)
        except Exception:
            pass

        out_rows: list[dict[str, Any]] = []
        for row in targets:
            api_path, warning = resolve_api_audio_path(row["abs_path"], media_root=media_root)
            if warning:
                notify_warning(warning)
            try:
                if action == "asr":
                    resp = await client.transcribe(
                        api_path,
                        model=settings["model"],
                        backend=settings["backend"],
                        device=settings["device"],
                        language=settings["language"],
                    )
                    segs = (resp.get("transcript") or {}).get("segments") or []
                    text = " ".join(s.get("text", "") for s in segs).strip()
                    out_rows.append({"path": row["path"], "ok": "✓" if text else "✗", "detail": text[:120]})
                elif action == "pipeline":
                    resp = await client.analyze_conversation(
                        api_path,
                        use_full_pipeline=True,
                        model=settings["model"],
                        backend=settings["backend"],
                        device=settings["device"],
                        language=settings["language"],
                    )
                    pipeline = resp.get("pipeline_results") or {}
                    qa = (pipeline.get("qa") or {}).get("overall_qa_score")
                    out_rows.append({"path": row["path"], "ok": "✓", "detail": f"QA={qa}"})
                else:
                    resp = await client.analyze_conversation(
                        api_path,
                        use_full_pipeline=False,
                        model=settings["model"],
                        backend=settings["backend"],
                        device=settings["device"],
                        language=settings["language"],
                    )
                    sents = resp.get("segment_sentiments") or []
                    label = sents[0].get("label") if sents else "?"
                    out_rows.append(
                        {
                            "path": row["path"],
                            "ok": "✓",
                            "detail": f"sentiment={label} (förv: {row.get('expected')})",
                        }
                    )
            except APIError as err:
                out_rows.append({"path": row["path"], "ok": "✗", "detail": str(err)})

        state.api_connected = bool(out_rows) and any(r["ok"] == "✓" for r in out_rows)
        results_holder["data"] = out_rows
        _render_results_panel(results_container, out_rows)
        notify_success(f"{action} via API klar")

    def _make_runner(action: str) -> Callable[[], Any]:
        async def _run() -> None:
            if run_mode.value == "API":
                await _run_api(action)
            else:
                await _run_local(action)

        return _run

    with ui.row().classes("gap-2 q-mb-md flex-wrap"):
        ui.button("Uppdatera lista", icon="refresh", on_click=_refresh_table)
        ui.button("Transkribera", icon="mic", on_click=_make_runner("asr"))
        ui.button("Pipeline", icon="call", on_click=_make_runner("pipeline"))
        ui.button("Sentimentkedja", icon="psychology", on_click=_make_runner("sentiment"))

    ui.separator().classes("q-my-md")
    render_section_title("Batch-scenario", icon="science")

    scenario_select = ui.select(
        label="Scenario",
        options={o["value"]: o["label"] for o in _SCENARIO_OPTIONS},
        value="smoke",
    ).classes("w-64")
    scenario_limit = ui.number(label="Max antal", value=3, min=1, max=50).classes("w-24")
    dry_run_cb = ui.checkbox("Dry-run (ingen ML)", value=False)

    async def run_scenario() -> None:
        if run_mode.value == "API":
            notify_warning("Batch-scenarier körs endast lokalt i v1")
            return
        results_container.clear()
        with results_container:
            ui.spinner(size="lg")
            ui.label("Kör scenario...").classes("text-caption")
        settings = _settings()
        try:
            report = await asyncio.to_thread(
                run_scenario_ui,
                scenario_select.value,
                pack_ids=[pack_select.value] if pack_select.value else None,
                emotions=[emotion_select.value] if (emotion_select.value or "") else None,
                limit=int(scenario_limit.value or 3),
                device=settings["device"],
                backend=settings["backend"],
                language=settings["language"],
                dry_run=bool(dry_run_cb.value),
            )
            results_holder["data"] = report
            _render_results_panel(results_container, report, title=f"Scenario: {scenario_select.value}")
            notify_success("Scenario klart")
        except Exception as exc:
            notify_error(str(exc))

    ui.button("Kör scenario", color="primary", icon="play_arrow", on_click=run_scenario).classes("q-mt-sm")

    ui.label(
        f"API-läge: sätt API_MEDIA_ROOT={repo_root()} och starta backend på port 8000."
    ).classes("text-caption text-grey q-mt-md")

    _refresh_table()


def _render_text_section(state: DashboardState) -> None:
    render_text_pipeline_section(state)

    ui.separator().classes("q-my-lg")
    render_section_title("Textsentiment (samples/examples.txt)", icon="text_fields")

    examples = load_examples_txt()
    if not examples:
        ui.label("Inga exempel hittades i samples/examples.txt").classes("text-caption")
        return

    example_opts = {t: t[:70] + ("…" if len(t) > 70 else "") for t in examples}
    selected_example = ui.select(
        label="Välj exempel",
        options=example_opts,
        value=examples[0],
    ).classes("w-full")
    custom_text = ui.textarea(label="Eller egen text", placeholder="Skriv text att analysera...").classes("w-full")
    text_result = ui.column().classes("w-full q-mt-md")

    async def run_text_sentiment() -> None:
        client = state.api_client
        if not client:
            notify_error("API-klient saknas — textsentiment kräver backend")
            return
        text = (custom_text.value or "").strip() or (selected_example.value or "").strip()
        if not text:
            notify_warning("Ange text")
            return
        text_result.clear()
        with text_result:
            ui.spinner(size="lg")
        try:
            if not state.api_connected:
                state.api_connected = await client.wait_for_health(attempts=3, interval=0.5)
            resp = await client.analyze_text([text], profile="default", return_all_scores=True)
            state.api_connected = True
            results = resp.get("results") or []
            text_result.clear()
            with text_result:
                if results:
                    first = results[0]
                    if isinstance(first, list) and first:
                        best = max(first, key=lambda x: float(x.get("score", 0)))
                        ui.label(f"Sentiment: {best.get('label')} (poäng {best.get('score', 0):.2f})").classes(
                            "text-body1"
                        )
                    else:
                        ui.label(f"Resultat: {first}").classes("text-body2")
                with ui.expansion("Fullständigt svar", icon="data_object"):
                    ui.code(json.dumps(resp, indent=2, ensure_ascii=False, default=str))
            notify_success("Textsentiment klart")
        except APIError as err:
            text_result.clear()
            with text_result:
                ui.label(str(err)).classes("text-negative")
            notify_api_error(err)

    ui.button("Analysera text", icon="sentiment_satisfied", on_click=run_text_sentiment).classes("q-mt-sm")


def _render_system_section(state: DashboardState) -> None:
    render_section_title("Systemhälsa", icon="health_and_safety")
    ui.label(f"Projektrot: {repo_root()}").classes("text-caption")
    media_root = os.environ.get("API_MEDIA_ROOT")
    if media_root:
        ui.label(f"API_MEDIA_ROOT: {media_root}").classes("text-caption text-positive")
    else:
        ui.label(
            f"API_MEDIA_ROOT ej satt — rekommenderat: {repo_root()}"
        ).classes("text-caption text-warning")

    if state.api_client:
        badge = "positive" if state.api_connected else "warning"
        ui.chip(
            f"Backend {state.api_client.base_url}: "
            + ("ansluten" if state.api_connected else "ej verifierad"),
            color=badge,
        ).classes("q-mb-sm")

    ui.label("Konfiguration hanteras i Setup Hub (Streamlit) eller launcher.").classes(
        "text-caption q-mb-md"
    )

    checks_container = ui.column().classes("w-full")

    async def run_doctor() -> None:
        checks_container.clear()
        with checks_container:
            ui.spinner(size="lg")
            ui.label("Kör hälsokontroll...").classes("text-caption")
        try:
            report = await asyncio.to_thread(run_doctor_check, require_openrouter=False)
            checks_container.clear()
            with checks_container:
                icon = "✅" if report.ok else "⚠️"
                ui.label(f"{icon} {'Alla kontroller OK' if report.ok else 'Vissa kontroller misslyckades'}").classes(
                    "text-subtitle2 q-mb-sm"
                )
                for check in report.checks:
                    mark = "✅" if check.ok else "❌"
                    ui.label(f"{mark} {check.name}: {check.message}").classes("text-body2")
                    if check.detail:
                        ui.label(check.detail).classes("text-caption text-grey q-ml-md")
            notify_success("Hälsokontroll klar")
        except Exception as exc:
            checks_container.clear()
            with checks_container:
                ui.label(f"Fel: {exc}").classes("text-negative")
            notify_error(str(exc))

    ui.button("Kör hälsokontroll", icon="medical_services", color="primary", on_click=run_doctor)


def _render_reports_section(results_holder: dict[str, Any]) -> None:
    render_section_title("Sparade ljudrapporter", icon="description")
    reports_container = ui.column().classes("w-full")
    preview_container = ui.column().classes("w-full q-mt-md")

    def refresh_reports() -> None:
        rows = list_audio_reports(limit=15)
        reports_container.clear()
        with reports_container:
            if not rows:
                ui.label("Inga reports/audio_*.json hittades.").classes("text-caption")
                return
            table = ui.table(
                columns=[
                    {"name": "file", "label": "Fil", "field": "file", "align": "left"},
                    {"name": "scenario", "label": "Scenario", "field": "scenario"},
                    {"name": "files", "label": "Filer", "field": "files"},
                    {"name": "accuracy", "label": "Sentiment-träff", "field": "accuracy"},
                    {"name": "success", "label": "OK", "field": "success"},
                ],
                rows=rows,
                row_key="id",
                selection="single",
            ).classes("w-full")

            async def on_select(event: dict) -> None:
                keys = event.get("rows") or []
                if not keys:
                    return
                match = next((r for r in rows if r["id"] in keys), None)
                if not match:
                    return
                try:
                    data = await asyncio.to_thread(load_report_json, match["path"])
                    preview_container.clear()
                    _render_results_panel(preview_container, data, title=match["file"])
                    if results_holder.get("container") is not None:
                        results_holder["data"] = data
                except Exception as exc:
                    notify_error(str(exc))

            table.on("selection", on_select)

    ui.button("Uppdatera listan", icon="refresh", on_click=refresh_reports).classes("q-mb-sm")
    refresh_reports()