"""Transcription Monitor UI – persistent queue, controls, live log.

Fas 3 WebSocket – docs/MIGRATION_TO_NICEGUI_PLAN.md §3
Push updates via TranscriptionState listeners; fallback ui.timer for queue drift.
"""

from __future__ import annotations

from nicegui import ui

from app.nicegui_dashboard.services.nicegui_api_client import NiceGUIAPIClient
from app.nicegui_dashboard.services.transcription_service import TranscriptionState


def render_transcription_tab(
    trans_state: TranscriptionState,
    *,
    api_client: NiceGUIAPIClient | None = None,
) -> None:
    """Render full transcription monitor view."""
    ui.label("🎙️ Transkriberingskö & Monitor").classes("text-h6")
    if api_client:
        ui.label(f"Backend: {api_client.base_url} | WebSocket live-loggar").classes("text-caption")
    else:
        ui.label("Persistent kö | Lokal fallback").classes("text-caption")

    status = trans_state.status
    pending = trans_state.scan_pending()

    metrics_row = ui.row().classes("w-full gap-4 q-mb-md")
    pending_metric = ui.metric("Väntande (persistent)", len(pending))
    status_metric = ui.metric("Status", "Pågår" if status.get("is_running") else "Idle")
    progress_metric = ui.metric("Progress", f"{status.get('processed', 0)}/{status.get('total', 0)}")
    api_metric = ui.metric("API Mode", "På" if status.get("use_api") else "Av")
    ws_metric = ui.metric("WebSocket", "—")

    progress_bar = ui.linear_progress(value=float(status.get("progress", 0))).classes("w-full q-mb-md")
    current_label = ui.label(f"Aktuell fil: {status.get('current_file') or '—'}").classes("text-caption")

    def _refresh_metrics() -> None:
        s = trans_state.status
        pending_metric.set_value(len(trans_state.queue))
        status_metric.set_value("Pågår" if s.get("is_running") else "Idle")
        progress_metric.set_value(f"{s.get('processed', 0)}/{s.get('total', 0)}")
        api_metric.set_value("På" if s.get("use_api") else "Av")
        ws_metric.set_value("Live" if trans_state.ws_connected else "Av")
        progress_bar.set_value(float(s.get("progress", 0)))
        current_label.set_text(f"Aktuell fil: {s.get('current_file') or '—'}")

    def start_batch() -> None:
        if trans_state.start_batch():
            ui.notify("Transkribering startad i bakgrunden")
            _refresh_metrics()
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

    with ui.row().classes("gap-2 q-mb-md"):
        ui.button("▶️ Start batch", color="primary", on_click=start_batch)
        ui.button("⏸️ Paus", on_click=pause_batch)
        ui.button("▶️ Återuppta", on_click=resume_batch)
        ui.button("⏹️ Stopp", color="negative", on_click=stop_batch)

    with ui.expander("⚙️ Inställningar"):
        settings = trans_state.settings

        def _set_backend(e) -> None:
            settings["backend"] = e.value

        def _set_preprocess(e) -> None:
            settings["preprocess"] = e.value

        def _set_diarize(e) -> None:
            settings["diarize"] = e.value

        def _set_hotwords(e) -> None:
            settings["hotwords"] = e.value or ""

        def _set_api(e) -> None:
            trans_state.status["use_api"] = bool(e.value)
            trans_state.save()

        ui.select(
            "Backend",
            options=["faster", "transformers", "whisperx"],
            value=settings.get("backend", "faster"),
            on_change=_set_backend,
        )
        ui.checkbox("Preprocess", value=settings.get("preprocess", True), on_change=_set_preprocess)
        ui.checkbox("Diarize", value=settings.get("diarize", False), on_change=_set_diarize)
        ui.textarea(
            "Hotwords",
            value=settings.get("hotwords", ""),
            on_change=_set_hotwords,
        )
        ui.checkbox(
            "Använd Backend API",
            value=status.get("use_api", False),
            on_change=_set_api,
        )

        def _set_strategy(e) -> None:
            trans_state.api_strategy = e.value or "transcribe"

        ui.select(
            "API-strategi",
            options=["transcribe", "batch_transcribe", "scan_process"],
            value=trans_state.api_strategy,
            on_change=_set_strategy,
        ).classes("w-full")

    ui.input(
        "Väntemapp",
        value=trans_state.pending_folder,
        on_change=lambda e: setattr(trans_state, "pending_folder", e.value or "inputs/pending"),
    ).classes("w-full q-mb-sm")

    ui.label("Väntande filer (persistent kö)").classes("text-subtitle2")
    queue_container = ui.column().classes("q-mb-md")

    def refresh_queue() -> None:
        queue_container.clear()
        with queue_container:
            files = trans_state.queue
            if not files:
                ui.label("Inga filer i kön.").classes("text-caption")
            else:
                for f in files[:8]:
                    ui.label(f"• {f.name}")
                if len(files) > 8:
                    ui.label(f"... + {len(files) - 8} till").classes("text-caption")

    refresh_queue()

    ui.label("📜 Händelselogg (live)").classes("text-subtitle2 q-mt-md")
    log_container = ui.column().classes("max-h-48 overflow-auto border q-pa-sm w-full")

    def refresh_log() -> None:
        log_container.clear()
        with log_container:
            for entry in trans_state.logs[-30:]:
                level = entry.get("level", "INFO")
                css = {
                    "INFO": "log-info",
                    "WARNING": "log-warning",
                    "ERROR": "log-error",
                }.get(level, "text-grey")
                file_part = f"{entry.get('file', '')}: " if entry.get("file") else ""
                src = entry.get("source", "")
                src_tag = " [ws]" if src == "ws" else ""
                ui.label(
                    f"{entry['ts']} [{level}]{src_tag} {file_part}{entry['msg']}"
                ).classes(css)

    refresh_log()

    def refresh_all() -> None:
        _refresh_metrics()
        refresh_log()
        refresh_queue()

    def on_push_event(event_type: str, _payload: dict) -> None:
        if event_type in {"log", "progress", "status", "done", "ws"}:
            refresh_all()

    trans_state.add_listener(on_push_event)

    # Fallback poll for queue file-system drift (WS handles log/progress push)
    ui.timer(5.0, refresh_queue)

    def clear_all() -> None:
        trans_state.clear_all()
        trans_state.scan_pending()
        refresh_all()
        ui.notify("Rensat kö och logg")

    ui.button("Rensa logg & kö", on_click=clear_all).classes("q-mt-md")