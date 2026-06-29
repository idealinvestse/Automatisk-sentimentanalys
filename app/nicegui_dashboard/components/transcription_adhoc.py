"""Ad-hoc single-file transcription UI for the Transkribering tab."""

from __future__ import annotations

import json
from typing import Any

from nicegui import ui

from app.nicegui_dashboard.services.nicegui_api_client import NiceGUIAPIClient
from app.nicegui_dashboard.services.transcription_presets import (
    DEFAULT_PRESET_ID,
    preset_description,
)
from app.nicegui_dashboard.services.transcription_result_helpers import (
    segments_table_rows,
    segments_to_csv_bytes,
    transcript_full_text,
    transcript_llm_enhanced,
    transcript_to_json_bytes,
)
from app.nicegui_dashboard.services.transcription_runtime import MAX_AUDIO_BYTES
from app.nicegui_dashboard.services.transcription_service import TranscriptionState

_AUDIO_ACCEPT = ".wav,.mp3,.m4a,.ogg,.flac"


def render_adhoc_section(
    trans_state: TranscriptionState,
    *,
    api_client: NiceGUIAPIClient | None = None,
) -> None:
    """Render quick ad-hoc transcription above the batch monitor."""
    _ = api_client  # reserved for future API-specific hints

    with ui.card().classes("w-full q-mb-md"):
        ui.label("Snabb transkribering").classes("text-subtitle1")
        preset_id = (
            trans_state.active_preset
            if trans_state.active_preset != "anpassad"
            else DEFAULT_PRESET_ID
        )
        ui.label(
            f"Ladda upp en ljudfil och få transkript direkt. "
            f"Aktiv preset: {preset_id} – {preset_description(preset_id)}"
        ).classes("text-caption q-mb-sm")
        ui.label(
            f"Max filstorlek: {MAX_AUDIO_BYTES // (1024 * 1024)} MB. "
            f"Tillåtna format: {_AUDIO_ACCEPT}"
        ).classes("text-caption text-grey q-mb-sm")

        file_label = ui.label("Ingen fil vald").classes("text-caption")

        status_row = ui.row().classes("w-full items-center gap-2 q-mt-sm")
        with status_row:
            spinner = ui.spinner(size="md")
            spinner.set_visibility(False)
            progress = ui.linear_progress(value=0.0).classes("flex-grow")
            progress.set_visibility(False)
            status_label = ui.label("").classes("text-caption")

        result_expansion = ui.expansion("Resultat", value=False).classes("w-full q-mt-sm")
        result_expansion.set_visibility(False)

        with result_expansion:
            ui.row().classes("gap-2 flex-wrap q-mb-sm")
            badge_asr = ui.badge("ASR", color="grey").classes("q-mr-xs")
            badge_llm = ui.badge("LLM-förbättrad", color="purple")
            badge_llm.set_visibility(False)
            meta_label = ui.label("").classes("text-caption")

            full_text = (
                ui.textarea(label="Fullständigt transkript", value="")
                .props("readonly outlined autogrow")
                .classes("w-full q-mb-sm")
            )

            segment_table = ui.table(
                columns=[
                    {"name": "time", "label": "Tid", "field": "time", "align": "left"},
                    {"name": "speaker", "label": "Talare", "field": "speaker", "align": "left"},
                    {"name": "text", "label": "Text", "field": "text", "align": "left"},
                    {
                        "name": "confidence",
                        "label": "Confidence",
                        "field": "confidence",
                        "align": "right",
                    },
                    {"name": "warning", "label": "Varning", "field": "warning", "align": "center"},
                ],
                rows=[],
                row_key="idx",
                pagination={"rowsPerPage": 15},
            ).classes("w-full q-mb-sm")

            export_holder: dict[str, Any] = {"transcript": None, "last_error": None}

            with ui.row().classes("gap-2 flex-wrap"):

                def copy_transcript() -> None:
                    text = full_text.value or ""
                    if not text:
                        ui.notify("Inget transkript att kopiera", type="warning")
                        return
                    ui.run_javascript(f"navigator.clipboard.writeText({json.dumps(text)})")
                    ui.notify("Transkript kopierat")

                def export_json() -> None:
                    tr = export_holder.get("transcript")
                    if not tr:
                        ui.notify("Inget resultat att exportera", type="warning")
                        return
                    ui.download(transcript_to_json_bytes(tr), "transkript.json")

                def export_csv() -> None:
                    tr = export_holder.get("transcript")
                    if not tr:
                        ui.notify("Inget resultat att exportera", type="warning")
                        return
                    ui.download(segments_to_csv_bytes(tr), "transkript.csv")

                ui.button("Kopiera transkript", on_click=copy_transcript).props("outline")
                ui.button("Exportera JSON", on_click=export_json).props("outline")
                ui.button("Exportera CSV", on_click=export_csv).props("outline")

        def _refresh_ui() -> None:
            running = trans_state.adhoc_running
            spinner.set_visibility(running)
            progress.set_visibility(running or trans_state.adhoc_progress > 0)
            progress.set_value(float(trans_state.adhoc_progress or 0))
            status_label.set_text(trans_state.adhoc_status_text or "")

            if trans_state.adhoc_filename:
                file_label.set_text(f"Vald fil: {trans_state.adhoc_filename}")
            elif not trans_state.adhoc_upload_path:
                file_label.set_text("Ingen fil vald")

            transcribe_btn.set_enabled(not running and bool(trans_state.adhoc_upload_path))
            cancel_btn.set_enabled(running)
            clear_btn.set_enabled(not running)

            if trans_state.adhoc_error and trans_state.adhoc_error != export_holder.get(
                "last_error"
            ):
                export_holder["last_error"] = trans_state.adhoc_error
                ui.notify(trans_state.adhoc_error, type="negative")
            if not trans_state.adhoc_error:
                export_holder["last_error"] = None

            transcript = trans_state.adhoc_result
            if transcript:
                result_expansion.set_visibility(True)
                result_expansion.value = True
                export_holder["transcript"] = transcript

                meta = trans_state.adhoc_meta or {}
                llm = transcript_llm_enhanced(transcript, meta)
                badge_llm.set_visibility(llm)
                badge_asr.set_visibility(not llm)
                meta_label.set_text(
                    f"Modell: {transcript.get('model', '—')} | "
                    f"Backend: {transcript.get('backend', '—')} | "
                    f"Längd: {transcript.get('duration', '—')}s | "
                    f"Bearbetning: {transcript.get('processing_time', '—')}s | "
                    f"Källa: {meta.get('source', '—')}"
                )
                full_text.set_value(transcript_full_text(transcript))
                segment_table.rows = segments_table_rows(transcript)
                segment_table.update()
            elif not running:
                result_expansion.set_visibility(bool(trans_state.adhoc_result))

        async def on_upload(e) -> None:
            try:
                content = e.content.read() if hasattr(e.content, "read") else e.content
                if isinstance(content, str):
                    content = content.encode("utf-8")
                name = getattr(e, "name", None) or "upload.wav"
                trans_state.save_adhoc_upload(content, name)
                ui.notify(f"Uppladdad: {name}")
                _refresh_ui()
            except Exception as err:
                ui.notify(f"Uppladdning misslyckades: {err}", type="negative")

        ui.upload(
            label="Välj fil",
            auto_upload=True,
            on_upload=on_upload,
        ).props(
            f'accept="{_AUDIO_ACCEPT}"'
        ).classes("w-full")

        with ui.row().classes("gap-2 q-mt-sm flex-wrap"):
            transcribe_btn = ui.button("Transkribera nu", color="primary")

            def start_transcription() -> None:
                if trans_state.start_adhoc_transcription():
                    ui.notify("Transkribering startad")
                    _refresh_ui()
                else:
                    ui.notify(
                        "Välj en fil först eller vänta tills pågående jobb är klart", type="warning"
                    )

            transcribe_btn.on_click(start_transcription)

            cancel_btn = ui.button("Avbryt", color="negative")

            def cancel_run() -> None:
                trans_state.cancel_adhoc()
                ui.notify("Avbryt begärd")
                _refresh_ui()

            cancel_btn.on_click(cancel_run)
            cancel_btn.set_enabled(False)

            clear_btn = ui.button("Rensa förhandsvisning")

            def clear_preview() -> None:
                trans_state.clear_adhoc_preview()
                result_expansion.set_visibility(False)
                file_label.set_text("Ingen fil vald")
                full_text.set_value("")
                segment_table.rows = []
                segment_table.update()
                ui.notify("Förhandsvisning rensad")

            clear_btn.on_click(clear_preview)

        def on_adhoc_event(event_type: str, _payload: dict) -> None:
            if event_type == "adhoc":
                _refresh_ui()

        trans_state.add_listener(on_adhoc_event)
        _refresh_ui()
