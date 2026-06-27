"""Transcription Monitor UI – persistent queue, controls, live log.

Fas 6.1 – docs/MIGRATION_TO_NICEGUI_PLAN.md (WebSocket reconnect + polling fallback)
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from nicegui import ui

from app.nicegui_dashboard.components.ui_primitives import metric_card, render_section_title
from app.nicegui_dashboard.components.transcription_adhoc import render_adhoc_section
from app.nicegui_dashboard.services.nicegui_api_client import NiceGUIAPIClient
from app.nicegui_dashboard.services.transcription_presets import (
    DEFAULT_PRESET_ID,
    apply_default_preset,
    apply_preset,
    is_recommended_preset,
    preset_description,
    preset_options,
)
from app.nicegui_dashboard.services.transcription_runtime import (
    DEFAULT_API_RETRIES,
    DEFAULT_LOCAL_TIMEOUT_S,
)
from app.nicegui_dashboard.services.transcription_service import TranscriptionState
from app.nicegui_dashboard.settings import ws_status_label


_LOG_LEVEL_SLOT = """
<q-td :props="props">
  <span :class="props.row.level_class">{{ props.value }}</span>
</q-td>
"""


def _log_level_class(level: str) -> str:
    lvl = str(level or "INFO").upper()
    if lvl == "ERROR":
        return "log-error"
    if lvl == "WARNING":
        return "log-warning"
    return "log-info"


def _fmt_elapsed(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    mins, secs = divmod(int(seconds), 60)
    if mins:
        return f"{mins}m {secs}s"
    return f"{secs}s"


def _log_level_counts(logs: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"ERROR": 0, "WARNING": 0, "INFO": 0}
    for entry in logs:
        lvl = str(entry.get("level", "INFO")).upper()
        if lvl in counts:
            counts[lvl] += 1
    return counts


def render_transcription_tab(
    trans_state: TranscriptionState,
    *,
    api_client: NiceGUIAPIClient | None = None,
    on_report_ready: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    """Render full transcription monitor view."""
    render_adhoc_section(trans_state, api_client=api_client)

    ui.label("🎙️ Transkriberingskö & Monitor").classes("text-h6")
    if api_client:
        ui.label(f"Backend: {api_client.base_url} | WebSocket live-loggar").classes("text-caption")
    else:
        ui.label("Persistent kö | Lokal fallback").classes("text-caption")

    status = trans_state.status
    settings = trans_state.settings
    scan_cfg = trans_state.scan_config

    render_section_title("Köstatus", icon="queue")
    with ui.row().classes("w-full gap-4 q-mb-md flex-wrap"):
        pending_metric = metric_card("Väntande", len(trans_state.queue))
        status_metric = metric_card("Status", "Pågår" if status.get("is_running") else "Vilande")
        progress_metric = metric_card("Framsteg", f"{status.get('processed', 0)}/{status.get('total', 0)}")
        api_metric = metric_card("API", "På" if status.get("use_api") else "Av")
        ws_metric = metric_card("WebSocket", ws_status_label(trans_state.ws_status))
        preset_metric = metric_card("Preset", trans_state.active_preset)

    render_section_title("Jobbinfo", icon="work")
    with ui.row().classes("w-full gap-4 q-mb-sm flex-wrap"):
        job_metric = metric_card("Job-ID", (status.get("active_job_id") or "—")[:8])
        elapsed_metric = metric_card("Tid", _fmt_elapsed(trans_state.elapsed_seconds()))

    render_section_title("Senaste jobb", icon="info")
    with ui.card().classes("w-full q-mb-sm"):
        last_job_label = ui.label("—").classes("text-body2")

    progress_bar = ui.linear_progress(value=float(status.get("progress", 0))).classes("w-full q-mb-sm")
    current_label = ui.label(f"Aktuell fil: {status.get('current_file') or '—'}").classes("text-caption")
    poll_hint = ui.label("").classes("text-caption text-warning")
    api_limit_hint = ui.label("").classes("text-caption text-grey")

    def _refresh_metrics() -> None:
        s = trans_state.status
        pending_metric.set_text(str(len(trans_state.queue)))
        status_metric.set_text("Pågår" if s.get("is_running") else "Vilande")
        progress_metric.set_text(f"{s.get('processed', 0)}/{s.get('total', 0)}")
        api_metric.set_text("På" if s.get("use_api") else "Av")
        ws_metric.set_text(ws_status_label(trans_state.ws_status))
        preset_metric.set_text(trans_state.active_preset)
        job_id = s.get("active_job_id") or ""
        job_metric.set_text(job_id[:8] if job_id else "—")
        elapsed_metric.set_text(_fmt_elapsed(trans_state.elapsed_seconds()))
        progress_bar.set_value(float(s.get("progress", 0)))
        current_label.set_text(f"Aktuell fil: {s.get('current_file') or '—'}")
        if trans_state.needs_polling_fallback():
            poll_hint.set_text("⚠ WebSocket nere – polling-fallback aktiv (2s)")
        else:
            poll_hint.set_text("")
        if s.get("use_api") and trans_state.api_strategy in ("batch_transcribe", "scan_process"):
            api_limit_hint.set_text(
                "ℹ️ Paus under pågående API-batch är begränsad – använd Avbryt för att stoppa backend-jobb."
            )
        else:
            api_limit_hint.set_text("")
        if trans_state.logs:
            last = trans_state.logs[-1]
            msg = str(last.get("msg", "—"))
            proc = s.get("processed", 0)
            total = s.get("total", 0)
            if total:
                msg = f"{msg} ({proc}/{total})"
            last_job_label.set_text(msg)
        else:
            last_job_label.set_text("Ingen logg ännu")

    def _update_poll_timer() -> None:
        poll_timer.active = trans_state.needs_polling_fallback()

    def start_batch() -> None:
        if trans_state.start_batch():
            ui.notify("Transkribering startad i bakgrunden")
            _refresh_metrics()
            _update_poll_timer()
        else:
            ui.notify("Inga filer i kön eller redan igång", type="warning")

    def pause_batch() -> None:
        trans_state.request_pause()
        ui.notify("Pausad")

    def resume_batch() -> None:
        trans_state.request_resume()
        ui.notify("Återupptagen")

    def stop_batch() -> None:
        trans_state.request_stop()
        ui.notify("Stopp begärd")

    async def cancel_batch() -> None:
        await trans_state.request_cancel()
        _refresh_metrics()
        ui.notify("Avbryt skickad")

    async def reconnect_ws() -> None:
        await trans_state.request_ws_reconnect()
        _refresh_metrics()
        _update_poll_timer()
        ui.notify("WebSocket reconnect initierad")

    def scan_folder() -> None:
        n = len(trans_state.scan_pending())
        refresh_queue()
        _refresh_metrics()
        ui.notify(f"Skannade mapp – {n} filer i kön")

    with ui.row().classes("gap-2 q-mb-md flex-wrap"):
        ui.button("Starta", icon="play_arrow", color="primary", on_click=start_batch)
        ui.button("Pausa", icon="pause", on_click=pause_batch)
        ui.button("Återuppta", icon="play_arrow", on_click=resume_batch)
        ui.button("Stoppa", icon="stop", color="negative", on_click=stop_batch)
        ui.button("Avbryt jobb", icon="cancel", color="negative", on_click=cancel_batch).props("outline")
        ui.button("Återanslut WS", icon="cable", on_click=reconnect_ws).props("outline")
        ui.button("Skanna mapp", icon="folder_open", on_click=scan_folder).props("outline")

    # --- Presets ---
    with ui.expansion("📋 Presets", value=False).classes("w-full"):
        preset_desc = ui.label("").classes("text-caption")

        def _on_preset_change(e) -> None:
            preset_desc.set_text(preset_description(e.value or ""))

        preset_select = ui.select(
            options=preset_options(),
            label="Välj preset",
            value=(
                trans_state.active_preset
                if trans_state.active_preset != "anpassad"
                else DEFAULT_PRESET_ID
            ),
            on_change=_on_preset_change,
        ).classes("w-full")
        preset_desc.set_text(preset_description(str(preset_select.value)))
        if is_recommended_preset(trans_state.active_preset):
            ui.label("★ Rekommenderad preset aktiv").classes("text-caption text-positive")

        def apply_selected_preset() -> None:
            pid = preset_select.value
            if not pid or pid == "anpassad":
                ui.notify("Välj en preset att tillämpa", type="warning")
                return
            if apply_preset(trans_state, pid):
                _sync_ui_from_state()
                refresh_all()
                ui.notify(f"Preset «{pid}» tillämpad")
            else:
                ui.notify("Okänd preset", type="negative")

        def reset_to_default_preset() -> None:
            if apply_default_preset(trans_state):
                preset_select.set_value(DEFAULT_PRESET_ID)
                preset_desc.set_text(preset_description(DEFAULT_PRESET_ID))
                _sync_ui_from_state()
                refresh_all()
                ui.notify("Återställt till callcenter standard")

        with ui.row().classes("gap-2 flex-wrap"):
            ui.button("Tillämpa preset", on_click=apply_selected_preset).props("outline")
            ui.button("Återställ till standard", on_click=reset_to_default_preset).props("outline")

    # --- ASR grund ---
    with ui.expansion("⚙️ ASR-grund", value=False).classes("w-full"):
        ui.select(
            options=["faster", "transformers", "whisperx"],
            label="Backend",
            value=settings.get("backend", "faster"),
            on_change=lambda e: trans_state.update_setting("backend", e.value),
        )
        ui.input(
            "Modell",
            value=settings.get("model", "kb-whisper-large"),
            on_change=lambda e: trans_state.update_setting("model", e.value or "kb-whisper-large"),
        )
        ui.select(
            options=["auto", "cpu", "cuda", "cuda:0", "mps"],
            label="Device",
            value=settings.get("device", "auto"),
            on_change=lambda e: trans_state.update_setting("device", e.value),
        )
        ui.input(
            "Språk",
            value=settings.get("language", "sv"),
            on_change=lambda e: trans_state.update_setting("language", e.value or "sv"),
        )
        ui.checkbox(
            "Preprocess",
            value=settings.get("preprocess", True),
            on_change=lambda e: trans_state.update_setting("preprocess", e.value),
        )
        ui.select(
            options=["standard", "strict", "subtitle"],
            label="Revision (valfri)",
            value=settings.get("revision") or "standard",
            on_change=lambda e: trans_state.update_setting(
                "revision", None if e.value == "standard" else e.value
            ),
        )
        def _set_use_api(e) -> None:
            trans_state.status["use_api"] = bool(e.value)
            trans_state.save()

        ui.checkbox(
            "Använd Backend API",
            value=status.get("use_api", False),
            on_change=_set_use_api,
        )

    # --- ASR avancerat ---
    with ui.expansion("🔧 ASR avancerat", value=False).classes("w-full"):
        ui.number(
            "Beam size",
            value=settings.get("beam_size", 5),
            min=1,
            max=10,
            on_change=lambda e: trans_state.update_setting("beam_size", int(e.value or 5)),
        )
        ui.checkbox(
            "VAD",
            value=settings.get("vad", True),
            on_change=lambda e: trans_state.update_setting("vad", e.value),
        )
        ui.number(
            "Chunk length (s)",
            value=settings.get("chunk_length_s", 30),
            min=5,
            max=60,
            on_change=lambda e: trans_state.update_setting("chunk_length_s", int(e.value or 30)),
        )
        ui.checkbox(
            "Diarize",
            value=settings.get("diarize", False),
            on_change=lambda e: trans_state.update_setting("diarize", e.value),
        )
        ui.number(
            "Antal talare",
            value=settings.get("num_speakers"),
            min=1,
            max=10,
            on_change=lambda e: trans_state.update_setting(
                "num_speakers", int(e.value) if e.value else None
            ),
        )
        ui.textarea(
            "Hotwords (kommaseparerade)",
            value=settings.get("hotwords", ""),
            on_change=lambda e: trans_state.update_setting("hotwords", e.value or ""),
        )
        ui.textarea(
            "Initial prompt",
            value=settings.get("initial_prompt") or "",
            on_change=lambda e: trans_state.update_setting("initial_prompt", e.value or None),
        )
        ui.checkbox(
            "Hotwords från fil",
            value=settings.get("use_hotwords_file", True),
            on_change=lambda e: trans_state.update_setting("use_hotwords_file", e.value),
        )
        ui.number(
            "Lokal timeout (s)",
            value=float(settings.get("local_timeout_s", DEFAULT_LOCAL_TIMEOUT_S)),
            min=60,
            max=7200,
            step=60,
            on_change=lambda e: trans_state.update_setting(
                "local_timeout_s", float(e.value or DEFAULT_LOCAL_TIMEOUT_S)
            ),
        )
        ui.number(
            "API-försök",
            value=int(settings.get("api_retries", DEFAULT_API_RETRIES)),
            min=1,
            max=5,
            on_change=lambda e: trans_state.update_setting("api_retries", int(e.value or DEFAULT_API_RETRIES)),
        )
        ui.checkbox(
            "API → lokal fallback",
            value=settings.get("api_fallback_local", True),
            on_change=lambda e: trans_state.update_setting("api_fallback_local", e.value),
        )

    # --- Batch & scan ---
    with ui.expansion("📦 Batch & scan", value=True).classes("w-full"):
        ui.select(
            options=["transcribe", "batch_transcribe", "scan_process"],
            label="API-strategi",
            value=trans_state.api_strategy,
            on_change=lambda e: (
                setattr(trans_state, "api_strategy", e.value or "transcribe"),
                trans_state.save(),
            ),
        ).classes("w-full")
        ui.number(
            "Workers",
            value=settings.get("workers", 1),
            min=1,
            max=8,
            on_change=lambda e: trans_state.update_setting("workers", int(e.value or 1)),
        )
        ui.number(
            "Worker timeout (s)",
            value=settings.get("worker_timeout", 300),
            min=30,
            max=3600,
            on_change=lambda e: trans_state.update_setting("worker_timeout", float(e.value or 300)),
        )
        ui.number(
            "Batch size (scan)",
            value=scan_cfg.get("batch_size", 4),
            min=1,
            max=64,
            on_change=lambda e: trans_state.update_setting("batch_size", int(e.value or 4), section="scan"),
        )
        ui.input(
            "Mönster (glob)",
            value=scan_cfg.get("pattern") or "",
            on_change=lambda e: trans_state.update_setting("pattern", e.value or None, section="scan"),
        )
        ui.number(
            "Max filer (scan)",
            value=scan_cfg.get("max_files"),
            min=1,
            on_change=lambda e: trans_state.update_setting(
                "max_files", int(e.value) if e.value else None, section="scan"
            ),
        )
        ui.checkbox(
            "Rekursiv scan",
            value=scan_cfg.get("recursive", True),
            on_change=lambda e: trans_state.update_setting("recursive", e.value, section="scan"),
        )
        ui.select(
            options=["transcribe", "analyze_conversation"],
            label="Operation (scan)",
            value=scan_cfg.get("operation", "transcribe"),
            on_change=lambda e: trans_state.update_setting("operation", e.value, section="scan"),
        )

    # --- Sökvägar ---
    with ui.expansion("📁 Sökvägar", value=False).classes("w-full"):
        ui.input(
            "Väntemapp",
            value=trans_state.pending_folder,
            on_change=lambda e: (
                setattr(trans_state, "pending_folder", e.value or "inputs/pending"),
                trans_state.save(),
            ),
        ).classes("w-full")
        ui.input(
            "Output-mapp",
            value=trans_state.output_dir,
            on_change=lambda e: (
                setattr(trans_state, "output_dir", e.value or "outputs/transcripts"),
                trans_state.save(),
            ),
        ).classes("w-full")

    with ui.card().classes("w-full q-mt-md"):
        render_section_title("Väntande filer (persistent kö)", icon="folder")
        queue_table = ui.table(
            columns=[
                {"name": "name", "label": "Fil", "field": "name", "align": "left"},
                {"name": "path", "label": "Sökväg", "field": "path", "align": "left"},
                {"name": "size_kb", "label": "KB", "field": "size_kb", "align": "right"},
                {"name": "modified", "label": "Ändrad", "field": "modified", "align": "left"},
            ],
            rows=[],
            row_key="path",
            selection="single",
            pagination={"rowsPerPage": 10},
        ).classes("w-full q-mb-sm")

        def refresh_queue() -> None:
            rows = trans_state.queue_rows()
            queue_table.rows = rows
            queue_table.update()

        def remove_selected_queue() -> None:
            selected = queue_table.selected
            if not selected:
                ui.notify("Välj en rad i kön", type="warning")
                return
            for row in selected:
                trans_state.remove_from_queue(row.get("path", ""))
            refresh_queue()
            ui.notify("Borttagen från kön")

        def export_queue() -> None:
            data = json.dumps(trans_state.queue_rows(), indent=2, ensure_ascii=False)
            ui.download(data.encode("utf-8"), "transcription_queue.json")

        with ui.row().classes("gap-2 q-mb-sm"):
            ui.button("Ta bort vald", on_click=remove_selected_queue).props("outline")
            ui.button("Rensa kö", on_click=lambda: (trans_state.clear_queue(), refresh_queue())).props("outline")
            ui.button("Exportera kö", on_click=export_queue).props("outline")

    with ui.card().classes("w-full q-mt-md"):
        render_section_title("Händelselogg (live)", icon="history")

        with ui.row().classes("w-full gap-4 flex-wrap q-mb-sm"):
            err_metric = metric_card("ERROR", 0, color="negative", size="compact")
            warn_metric = metric_card("WARNING", 0, color="warning", size="compact")
            info_metric = metric_card("INFO", 0, color="primary", size="compact")

        log_table = ui.table(
            columns=[
                {"name": "ts", "label": "Tid", "field": "ts", "align": "left"},
                {"name": "level", "label": "Nivå", "field": "level", "align": "left"},
                {"name": "source", "label": "Källa", "field": "source", "align": "left"},
                {"name": "file", "label": "Fil", "field": "file", "align": "left"},
                {"name": "msg", "label": "Meddelande", "field": "msg", "align": "left"},
            ],
            rows=[],
            row_key="ts",
            pagination={"rowsPerPage": 0},
        ).classes("w-full max-h-64")
        log_table.add_slot("body-cell-level", _LOG_LEVEL_SLOT)

        log_filters: dict[str, Any] = {"level": "", "source": "", "search": "", "limit": 100}

        def refresh_log() -> None:
            entries = trans_state.filtered_logs(
                level=log_filters["level"] or None,
                source=log_filters["source"] or None,
                search=log_filters["search"] or "",
                limit=int(log_filters["limit"] or 100),
            )
            log_table.rows = [
                {
                    "ts": e.get("ts", ""),
                    "level": e.get("level", "INFO"),
                    "level_class": _log_level_class(e.get("level", "INFO")),
                    "source": e.get("source", "local"),
                    "file": e.get("file") or "",
                    "msg": e.get("msg", ""),
                }
                for e in entries
            ]
            log_table.update()
            counts = _log_level_counts(trans_state.logs)
            err_metric.set_text(str(counts["ERROR"]))
            warn_metric.set_text(str(counts["WARNING"]))
            info_metric.set_text(str(counts["INFO"]))

        with ui.row().classes("w-full gap-2 flex-wrap q-mb-sm"):
            ui.select(
                options={"": "Alla", "INFO": "INFO", "WARNING": "WARNING", "ERROR": "ERROR"},
                label="Nivå",
                value="",
                on_change=lambda e: (log_filters.update({"level": e.value or ""}), refresh_log()),
            ).classes("w-32")
            ui.select(
                options={"": "Alla", "local": "Lokal", "ws": "WebSocket"},
                label="Källa",
                value="",
                on_change=lambda e: (log_filters.update({"source": e.value or ""}), refresh_log()),
            ).classes("w-32")
            ui.input(
                "Sök",
                value="",
                on_change=lambda e: (log_filters.update({"search": e.value or ""}), refresh_log()),
            ).classes("flex-grow")
            ui.select(
                options={50: "50", 100: "100", 200: "200"},
                label="Visa",
                value=100,
                on_change=lambda e: (log_filters.update({"limit": e.value or 100}), refresh_log()),
            ).classes("w-24")

        def export_log() -> None:
            data = json.dumps(trans_state.logs, indent=2, ensure_ascii=False)
            ui.download(data.encode("utf-8"), f"transcription_log_{datetime.now():%Y%m%d_%H%M%S}.json")

        with ui.row().classes("gap-2 q-mt-sm"):
            ui.button("Rensa logg", on_click=lambda: (trans_state.clear_logs(), refresh_log())).props("outline")
            ui.button("Exportera logg", on_click=export_log).props("outline")

    def _sync_ui_from_state() -> None:
        nonlocal settings, scan_cfg, status
        settings = trans_state.settings
        scan_cfg = trans_state.scan_config
        status = trans_state.status

    def refresh_all() -> None:
        _refresh_metrics()
        refresh_log()
        refresh_queue()
        _update_poll_timer()

    last_reported: dict[str, Any] = {"key": None}

    def on_push_event(event_type: str, payload: dict) -> None:
        if event_type in {"log", "progress", "status", "done", "ws", "log_clear", "queue", "adhoc"}:
            refresh_all()
        if on_report_ready and event_type == "adhoc":
            result = payload.get("result")
            if result and not payload.get("running"):
                report_key = (
                    payload.get("filename")
                    or str(id(result))
                )
                if last_reported["key"] != report_key:
                    last_reported["key"] = report_key
                    from app.nicegui_dashboard.services.report_ingest import (
                        normalize_transcription_to_report,
                    )

                    stem = Path(str(payload.get("filename") or "")).stem or None
                    report = normalize_transcription_to_report(result, call_id=stem)
                    on_report_ready(report)

    trans_state.add_listener(on_push_event)

    refresh_queue()
    refresh_log()
    _refresh_metrics()

    poll_timer = ui.timer(2.0, refresh_all, active=False)
    ui.timer(5.0, refresh_queue)
    ui.timer(1.0, lambda: elapsed_metric.set_text(_fmt_elapsed(trans_state.elapsed_seconds())))

    def clear_everything() -> None:
        trans_state.clear_all()
        trans_state.scan_pending()
        refresh_all()
        ui.notify("Rensat kö och logg")

    ui.button("Rensa allt (kö + logg + status)", on_click=clear_everything).classes("q-mt-md").props("outline")