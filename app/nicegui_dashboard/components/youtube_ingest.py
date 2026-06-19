"""YouTube Ingest tab – Download audio from YouTube for test data.

Integrates with the new /ingest/youtube API endpoints.
Fas 4 implementation.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable

from nicegui import ui

from app.nicegui_dashboard.services.nicegui_api_client import APIError, NiceGUIAPIClient
from app.nicegui_dashboard.services.ui_helpers import notify_error, notify_success, notify_warning
from app.nicegui_dashboard.state import DashboardState


def render_youtube_ingest_tab(state: DashboardState) -> None:
    """Render YouTube data ingestion section."""
    ui.label("YouTube Datainsamling").classes("text-h5 q-mb-sm")
    ui.label(
        "Ladda ner ljud från YouTube för att bygga svenska testdataset till ASR och sentimentanalys. "
        "Endast offentligt material för testbruk."
    ).classes("text-caption q-mb-md")

    # Form section
    with ui.card().classes("w-full q-pa-md q-mb-md"):
        ui.label("Ny nedladdning").classes("text-subtitle1 q-mb-sm")

        url_input = ui.input(
            label="YouTube URL eller spellistlänk",
            placeholder="https://www.youtube.com/watch?v=... eller playlist",
        ).classes("w-full")

        with ui.row().classes("gap-4 flex-wrap q-mb-sm"):
            playlist_cb = ui.checkbox("Spellista", value=False)
            wav_cb = ui.checkbox("Konvertera till 16kHz mono WAV", value=True)
            auto_transcribe_cb = ui.checkbox("Auto-transkribera efter nedladdning", value=False)
            auto_analyze_cb = ui.checkbox("Auto-analysera efter nedladdning", value=False)

        progress_bar = ui.linear_progress(value=0, show_value=False).classes("w-full q-mb-sm")
        status_label = ui.label("").classes("text-caption")

        async def start_download():
            url = (url_input.value or "").strip()
            if not url:
                notify_warning("Ange en YouTube-URL")
                return

            progress_bar.value = 0
            status_label.text = "Startar nedladdning..."
            status_label.classes("text-positive")

            client: NiceGUIAPIClient | None = getattr(state, "api_client", None)
            if not client:
                notify_error("API-klient saknas. Starta backend först.")
                return

            try:
                # Call the new ingest endpoint
                payload = {
                    "url": url,
                    "playlist": playlist_cb.value,
                    "convert_to_wav": wav_cb.value,
                    "auto_transcribe": auto_transcribe_cb.value,
                    "auto_analyze": auto_analyze_cb.value,
                }

                # Simulate progress (real progress would require WebSocket or polling)
                for i in range(1, 6):
                    await asyncio.sleep(0.3)
                    progress_bar.value = i / 5

                resp = await client.download_youtube(
                    url,
                    playlist=payload["playlist"],
                    convert_to_wav=payload["convert_to_wav"],
                    auto_transcribe=payload["auto_transcribe"],
                    auto_analyze=payload["auto_analyze"],
                )

                progress_bar.value = 1.0

                if resp.get("success"):
                    status_label.text = resp.get("message", "Nedladdning klar!")
                    status_label.classes("text-positive")
                    notify_success("YouTube-ljud nedladdat!")
                    # Refresh the list
                    await refresh_list()
                else:
                    status_label.text = f"Fel: {resp.get('error', 'Okänt fel')}"
                    status_label.classes("text-negative")
                    notify_error(resp.get("error", "Nedladdning misslyckades"))

            except APIError as err:
                status_label.text = f"API-fel: {err}"
                status_label.classes("text-negative")
                notify_error(str(err))
            except Exception as exc:
                status_label.text = f"Fel: {exc}"
                status_label.classes("text-negative")
                notify_error(str(exc))
            finally:
                # Reset progress after a moment
                await asyncio.sleep(1.5)
                progress_bar.value = 0
                status_label.text = ""

        ui.button(
            "Ladda ner från YouTube",
            icon="download",
            color="primary",
            on_click=start_download,
        ).classes("q-mt-sm")

    # Ingested files list
    ui.label("Tidigare nedladdade filer").classes("text-subtitle1 q-mb-sm")

    list_container = ui.column().classes("w-full")

    async def refresh_list():
        list_container.clear()
        client: NiceGUIAPIClient | None = getattr(state, "api_client", None)
        if not client:
            with list_container:
                ui.label("API-klient saknas").classes("text-negative")
            return

        try:
            files = await client.list_youtube_ingested(limit=50)
            if not files:
                with list_container:
                    ui.label("Inga filer ännu. Ladda ner något ovanför!").classes("text-caption")
                return

            with list_container:
                columns = [
                    {"name": "title", "label": "Titel", "field": "title", "align": "left"},
                    {"name": "duration", "label": "Längd (s)", "field": "duration_seconds"},
                    {"name": "format", "label": "Format", "field": "metadata.format"},
                    {"name": "actions", "label": "Åtgärder", "field": "actions"},
                ]

                def make_actions(row: dict) -> None:
                    with ui.row().classes("gap-1"):
                        ui.button("Transkribera", icon="mic", size="sm", on_click=lambda: trigger_transcribe(row)).classes("text-xs")
                        ui.button("Analysera", icon="analytics", size="sm", on_click=lambda: trigger_analyze(row)).classes("text-xs")
                        ui.button("Ta bort", icon="delete", color="negative", size="sm", on_click=lambda: delete_file(row)).classes("text-xs")

                rows = []
                for f in files:
                    meta = f.get("metadata", {})
                    rows.append({
                        "title": f.get("title", "?")[:60],
                        "duration_seconds": meta.get("duration_seconds") or f.get("duration_seconds"),
                        "metadata": meta,
                        "file_path": f.get("file_path"),
                        "actions": "",  # Will be rendered in slot
                    })

                table = ui.table(
                    columns=columns,
                    rows=rows,
                    row_key="file_path",
                ).classes("w-full")

                # Custom actions column (NiceGUI table slot)
                table.add_slot("actions", make_actions)

        except APIError as err:
            with list_container:
                ui.label(f"Kunde inte ladda lista: {err}").classes("text-negative")

    async def trigger_transcribe(row: dict):
        file_path = row.get("file_path")
        if not file_path:
            return
        notify_warning("Transkribering från ingest UI kommer i nästa iteration (använd CLI/API för nu)")
        # TODO: Call /transcribe with the file_path

    async def trigger_analyze(row: dict):
        file_path = row.get("file_path")
        if not file_path:
            return
        notify_warning("Analys från ingest UI kommer i nästa iteration")
        # TODO: Call analyze pipeline

    async def delete_file(row: dict):
        youtube_id = None
        meta = row.get("metadata", {})
        if "youtube_id" in meta:
            youtube_id = meta["youtube_id"]
        elif row.get("file_path"):
            # Extract from filename if possible
            pass

        if not youtube_id:
            notify_warning("Kunde inte identifiera YouTube ID för borttagning")
            return

        client = state.api_client
        if not client:
            return

        try:
            await client.delete_youtube_ingested(youtube_id)
            notify_success("Fil borttagen")
            await refresh_list()
        except Exception as exc:
            notify_error(str(exc))

    # Initial load button
    with ui.row().classes("q-mb-md"):
        ui.button("Uppdatera lista", icon="refresh", on_click=refresh_list)

    # Load list on render
    ui.timer(0.1, refresh_list, once=True)
