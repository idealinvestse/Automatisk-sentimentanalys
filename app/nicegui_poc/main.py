"""
NiceGUI Proof-of-Concept Dashboard for Automatisk Sentimentanalys (Fas 5.1)

Replikerar nyckelfunktioner från den befintliga Streamlit-dashboarden:
- Översikt med KPI-kort, trender, hot topics, agent leaderboard, alerts
- Call Detail View (header, timeline, transcript, structured insights)
- Transkriberings Monitor (persistent kö, progress, start/paus/stopp, inställningar, logg, API-toggle)
- Live-analys

Kör med:
    pip install nicegui httpx pandas plotly
    python -m app.nicegui_poc.main

Designat för att visa hur NiceGUI kan ersätta/komplettera Streamlit med bättre reaktivitet och modern UX.
Integration med befintlig FastAPI-backend är markerad med TODOs.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from nicegui import ui, app
except ImportError:
    print("Install nicegui: pip install nicegui")
    raise

# Simulerad data från befintlig pipeline/demo (ersätt med riktiga API-anrop)
DEMO_REPORTS = [
    {
        "call_id": "CALL-001",
        "title": "Faktura fel – lyckad upplösning",
        "meta": {"agent": "Agent-Anna", "duration_s": 420},
        "overall_sentiment": "positiv",
        "qa_score": 92,
        "risk_level": "low",
        "alerts_count": 0,
    },
    {
        "call_id": "CALL-002",
        "title": "Lång väntetid + arg kund",
        "meta": {"agent": "Agent-Bengt", "duration_s": 310},
        "overall_sentiment": "negativ",
        "qa_score": 65,
        "risk_level": "high",
        "alerts_count": 2,
    },
]

TRANSCRIPTION_LOGS: list[dict] = []
TRANSCRIPTION_STATUS = {
    "is_running": False,
    "current_file": None,
    "progress": 0.0,
    "processed": 0,
    "total": 0,
}
PENDING_QUEUE: list[str] = []


def add_log(level: str, msg: str, file: str | None = None):
    TRANSCRIPTION_LOGS.append({
        "ts": datetime.now().isoformat(timespec="seconds"),
        "level": level,
        "msg": msg,
        "file": file,
    })
    if len(TRANSCRIPTION_LOGS) > 100:
        TRANSCRIPTION_LOGS[:] = TRANSCRIPTION_LOGS[-80:]


@ui.page("/")
def main_page():
    ui.page_title("Call Center Insights – NiceGUI PoC (Fas 5.1)")
    ui.dark_mode().enable()

    with ui.header(elevated=True).classes("items-center justify-between"):
        ui.label("📞 Svensk Call Center – Samtalsintelligens").classes("text-h5")
        ui.label("NiceGUI PoC | Streamlit → NiceGUI migration").classes("text-caption")

    with ui.tabs() as tabs:
        overview_tab = ui.tab("Översikt")
        detail_tab = ui.tab("Call Detail")
        trans_tab = ui.tab("Transkribering")
        live_tab = ui.tab("Live-analys")

    with ui.tab_panels(tabs, value=overview_tab).classes("w-full"):
        # ==================== ÖVERSIKT ====================
        with ui.tab_panel(overview_tab):
            ui.label("📊 Översikt – KPI:er & Filter").classes("text-h6 q-mb-md")

            # KPI Cards
            with ui.row().classes("w-full gap-4"):
                for label, value, color in [
                    ("Totalt samtal", "5", "primary"),
                    ("Positiva", "60%", "positive"),
                    ("Negativa", "25%", "negative"),
                    ("QA Snitt", "78/100", "warning"),
                    ("Alerts", "3", "negative"),
                ]:
                    with ui.card().classes("flex-1"):
                        ui.label(label).classes("text-caption")
                        ui.label(value).classes(f"text-h4 text-{color}")

            ui.separator()

            # Hot Topics & Agent Leaderboard
            with ui.row().classes("w-full gap-4"):
                with ui.card().classes("flex-1"):
                    ui.label("🔥 Hot Topics").classes("text-subtitle2")
                    for topic in ["Fakturering", "Teknisk support", "Väntetid", "Empati"]:
                        ui.chip(topic, color="primary").classes("q-ma-xs")

                with ui.card().classes("flex-1"):
                    ui.label("👥 Agent Leaderboard").classes("text-subtitle2")
                    ui.table(
                        columns=[{"name": "agent", "label": "Agent", "field": "agent"},
                                 {"name": "calls", "label": "Samtal", "field": "calls"},
                                 {"name": "qa", "label": "QA", "field": "qa"}],
                        rows=[
                            {"agent": "Agent-Anna", "calls": 42, "qa": "92"},
                            {"agent": "Agent-Erika", "calls": 38, "qa": "85"},
                        ],
                        row_key="agent"
                    ).classes("w-full")

            # Filtered calls table
            ui.label("📋 Senaste samtal (klicka för detalj)").classes("text-subtitle2 q-mt-md")
            table = ui.table(
                columns=[
                    {"name": "call_id", "label": "ID", "field": "call_id"},
                    {"name": "title", "label": "Ämne", "field": "title"},
                    {"name": "agent", "label": "Agent", "field": "meta.agent"},
                    {"name": "sentiment", "label": "Sentiment", "field": "overall_sentiment"},
                    {"name": "qa", "label": "QA", "field": "qa_score"},
                ],
                rows=DEMO_REPORTS,
                row_key="call_id",
            ).classes("w-full")

            def open_detail(e):
                selected = e.args[0]  # rough extraction
                ui.notify(f"Öppnar detalj för {selected.get('call_id', 'CALL')}")
                # I riktig implementation: byt tab + ladda data
                tabs.set_value(detail_tab)

            # Simple click simulation
            ui.button("Visa exempel Call Detail", on_click=lambda: tabs.set_value(detail_tab)).classes("q-mt-md")

        # ==================== CALL DETAIL ====================
        with ui.tab_panel(detail_tab):
            ui.label("🔍 Call Detail View – Exempel CALL-001").classes("text-h6")

            with ui.card():
                ui.label("Header / Meta").classes("text-subtitle2")
                with ui.row():
                    ui.label("Agent: Agent-Anna | Duration: 7 min | Sentiment: Positiv | QA: 92/100")

            ui.separator()

            ui.label("Interaktiv Timeline (simulerad)").classes("text-subtitle2 q-mt-md")
            with ui.card():
                ui.label("[00:00-00:08] Agent: Hej, hur kan jag hjälpa dig idag?")
                ui.label("[00:08-00:32] Kund: Jag har problem med fakturan...")
                ui.label("... (klickbara segment i full version)")

            ui.label("Transkript (sökbart + highlights)").classes("text-subtitle2 q-mt-md")
            with ui.card().classes("max-h-64 overflow-auto"):
                ui.textarea("Sök i transkript...", value="faktura").classes("w-full")
                ui.markdown("""
**Agent:** Tack för att du ringer. Jag förstår att det känns frustrerande.
**Kund:** Ja, det är skandal!
                """).classes("q-pa-md")

            ui.label("Structured Insights (LLM + Fas4)").classes("text-subtitle2 q-mt-md")
            with ui.expander("Actionable Summary & Agent Assessment", expanded=True):
                ui.markdown("""
**Problem:** Feldebitering pga systemfel
**Rekommendationer:**
- Kreditera 890 kr
- Skicka rättad faktura
- Notera i kundprofil

**Empati-score:** 0.92
**Styrkor:** Snabb problemlösning, vänlig ton
                """)

            ui.button("Tillbaka till Översikt", on_click=lambda: tabs.set_value(overview_tab))

        # ==================== TRANSKRIBERING ====================
        with ui.tab_panel(trans_tab):
            ui.label("🎙️ Transkriberingskö & Monitor (NiceGUI PoC)").classes("text-h6")
            ui.label("Persistent kö + API-polling redo | Icke-blockerande").classes("text-caption")

            # Status KPIs
            with ui.row().classes("w-full gap-4 q-mb-md"):
                ui.metric("Väntande (persistent)", len(PENDING_QUEUE) or 12)
                ui.metric("Status", "Pågår" if TRANSCRIPTION_STATUS["is_running"] else "Idle")
                ui.metric("Progress", f"{TRANSCRIPTION_STATUS['processed']}/{TRANSCRIPTION_STATUS['total']}")
                ui.metric("API Mode", "På" if TRANSCRIPTION_STATUS.get("use_api") else "Av")

            # Controls
            with ui.row().classes("gap-2"):
                ui.button("▶️ Start batch", color="primary", on_click=start_transcription)
                ui.button("⏸️ Paus", on_click=pause_transcription)
                ui.button("▶️ Återuppta", on_click=resume_transcription)
                ui.button("⏹️ Stopp", color="negative", on_click=stop_transcription)

            # Settings
            with ui.expander("⚙️ Inställningar", expanded=False):
                ui.select("Backend", options=["faster", "transformers", "whisperx"], value="faster")
                ui.checkbox("Preprocess", value=True)
                ui.checkbox("Diarize", value=False)
                ui.textarea("Hotwords", value="fakturering,återbetalning")
                use_api = ui.checkbox("Använd Backend API (/scan_process)", value=TRANSCRIPTION_STATUS.get("use_api", False))

            # Log
            ui.label("📜 Händelselogg").classes("text-subtitle2 q-mt-md")
            log_container = ui.column().classes("max-h-48 overflow-auto border q-pa-sm")

            def refresh_log():
                log_container.clear()
                with log_container:
                    for entry in TRANSCRIPTION_LOGS[-30:]:
                        color = {"INFO": "green", "WARNING": "orange", "ERROR": "red"}.get(entry["level"], "grey")
                        ui.label(f"{entry['ts']} [{entry['level']}] {entry.get('file','')}: {entry['msg']}").classes(f"text-{color}")

            ui.timer(2.0, refresh_log)  # live update

            ui.button("Rensa logg & kö", on_click=lambda: (TRANSCRIPTION_LOGS.clear(), PENDING_QUEUE.clear(), ui.notify("Rensat")))

        # ==================== LIVE-ANALYS ====================
        with ui.tab_panel(live_tab):
            ui.label("🧪 Live-analys (simulerad)").classes("text-h6")
            ui.textarea("Klistra in segments (JSON)", placeholder='[{"text": "Hej", "speaker": "Agent"}]').classes("w-full")
            ui.button("Analysera (pipeline)", on_click=lambda: ui.notify("Kör pipeline via backend (TODO: httpx till /analyze_pipeline)"))

            ui.label("Resultat visas här i full implementation").classes("text-caption q-mt-md")


def start_transcription():
    if TRANSCRIPTION_STATUS["is_running"]:
        return
    TRANSCRIPTION_STATUS["is_running"] = True
    TRANSCRIPTION_STATUS["total"] = 5
    TRANSCRIPTION_STATUS["processed"] = 0
    add_log("INFO", "Startade batch (NiceGUI PoC)")
    ui.notify("Transkribering startad i bakgrunden (simulerad)")

    # Simulerad bakgrundsprocess
    async def run():
        for i in range(5):
            if not TRANSCRIPTION_STATUS["is_running"]:
                break
            await asyncio.sleep(1.5)
            TRANSCRIPTION_STATUS["processed"] = i + 1
            TRANSCRIPTION_STATUS["progress"] = (i + 1) / 5
            add_log("INFO", f"Klart fil {i+1}", file=f"call_{i+1}.wav")
        TRANSCRIPTION_STATUS["is_running"] = False
        add_log("INFO", "Batch slutförd")

    asyncio.create_task(run())


def pause_transcription():
    TRANSCRIPTION_STATUS["is_running"] = False
    add_log("WARNING", "Paus begärd")
    ui.notify("Pausad")


def resume_transcription():
    TRANSCRIPTION_STATUS["is_running"] = True
    add_log("INFO", "Återupptagen")
    ui.notify("Återupptagen")


def stop_transcription():
    TRANSCRIPTION_STATUS["is_running"] = False
    TRANSCRIPTION_STATUS["processed"] = 0
    add_log("WARNING", "Stopp begärd")
    ui.notify("Stoppad")


if __name__ in {"__main__", "__mp_main__"}:
    # Kör PoC
    ui.run(
        host="0.0.0.0",
        port=8080,
        title="Call Center NiceGUI PoC",
        reload=False,
    )
