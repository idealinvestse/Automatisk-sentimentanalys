"""
MIGRATION TO NICEGUI – Detaljerad Projektplan & Guide
Automatisk Sentimentanalys (Call Center Intelligence)

Version: 1.0
Datum: 2026-06-13
Mål: Migrera befintlig Streamlit-dashboard till NiceGUI för bättre prestanda, reaktivitet, modern UX och långsiktig skalbarhet.

Baserat på:
- Befintlig Streamlit-dashboard (app/dashboard.py + components/)
- NiceGUI PoC (app/nicegui_poc/main.py)
- Befintlig FastAPI-backend
- Transkriberings Monitor med persistent kö + API-polling
"""

# 1. Varför migrera från Streamlit till NiceGUI?

## Fördelar med NiceGUI för detta projekt
- **Bättre reaktivitet**: Event-driven istället för fulla script-reruns → snabbare UI-uppdateringar (viktigt för transkriberings-progress och live logs).
- **Modern UX**: Tailwind-stöd, bättre komponenter, desktop-liknande känsla.
- **Bättre state management**: Explicit hantering av session/state, enklare multi-page och komplexa vyer.
- **Prestanda**: Mindre onödig kodexekvering, bättre skalbarhet för call center-data.
- **Integration med backend**: Bygger naturligt på FastAPI (ni har redan en stark backend).
- **Framtidssäkerhet**: Lättare att lägga till WebSocket, real-time features, autentisering.
- **Community & momentum 2026**: Många rekommenderar NiceGUI för interna data/AI-verktyg.

## Nackdelar / Risker
- Lite längre inlärningskurva än Streamlit (men fortfarande Python-only).
- Mindre "batteries-included" för vissa data-viz (men Plotly-integration finns).
- Mindre mogen än Streamlit för vissa edge cases.

**Slutsats**: NiceGUI är det bästa nästa steget för ert projekt – det ger mest värde per investerad tid.

# 2. Övergripande Migreringsstrategi

## Fasindelning (rekommenderad)

### Fas 1: Foundation & PoC-validering (1–2 veckor)
- Sätta upp NiceGUI-projektstruktur
- Migrera grundläggande layout + navigation (tabs/pages)
- Portning av KPI-kort och enkla tabeller
- Validera PoC mot befintlig funktionalitet

### Fas 2: Core Views (3–4 veckor)
- Översikt (KPI, Hot Topics, Agent Leaderboard, filtrerad tabell)
- Call Detail View (header, timeline, transcript, structured insights)
- Transkriberings Monitor (full persistent queue, controls, settings, live log)

### Fas 3: Avancerad funktionalitet & Backend-integration (2–3 veckor)
- Live-analys + pipeline-anrop via httpx
- Real API-polling mot `/scan_process` och `/batch_transcribe`
- WebSocket-stöd för real-time transkriberingsstatus (valfritt men starkt rekommenderat)
- State-persistens och bättre felhantering

### Fas 4: Polering, Test & Deployment (1–2 veckor)
- UI/UX-förbättringar, teman, responsivitet
- Enhetstester + E2E-tester
- Dokumentation & migreringsguide för teamet
- Deployment (Docker + uvicorn + NiceGUI)

### Fas 5: Avveckling av Streamlit (1 vecka)
- Parallellkörning (feature flag)
- Full övergång
- Ta bort gammal kod

**Total estimat**: 8–12 veckor (kan parallelliseras delvis)

# 3. Detaljerad Task Breakdown

## Fas 1 Tasks
- [x] Skapa `app/nicegui_dashboard/` struktur (main.py, components/, services/)
- [x] Grundläggande layout med ui.header, ui.tabs, ui.tab_panels
- [x] Portning av KPI-metrics (ui.metric eller ui.card)
- [x] Enkel tabell-komponent (ui.table)
- [x] Validering att PoC matchar Streamlit-funktionalitet

### Fas 1 Status: ✅ Klar (2026-06-13)
- **Struktur**: `app/nicegui_dashboard/` (modulär: `components/`, `services/`, `state.py`)
- **Kör**: `pip install -e ".[dashboard-nicegui]"` → `python -m app.nicegui_dashboard.main`
- **Levererat**: dark mode header, 4 flikar (Översikt full, övriga stubs), KPI via `compute_kpis`, samtalstabell med radklick, demo-data via `demo_provider.py` + `data_services`
- **Nästa**: Fas 2 (Core Views)

## Fas 2 Tasks
- [x] Översiktsvy: Hot Topics chips, Agent Leaderboard tabell, filtrerad calls-tabell med on_click
- [x] Call Detail: ui.card för header, timeline (ui.timeline eller custom), sökbart transcript (ui.textarea + markdown)
- [x] Structured Insights: ui.expander + ui.markdown för LLM + Fas4-data
- [x] Transkriberings Monitor: Full implementation av persistent JSON-kö, start/paus/stopp, live progress, loggruta, settings-form

### Fas 2 Status: ✅ Klar (2026-06-13)
- **Översikt**: Sentiment/agent/sök-filter med `@ui.refreshable`, KPI + tabell uppdateras reaktivt
- **Call Detail**: Header, klickbar timeline, sökbart transkript, structured insights (LLM/Fas4)
- **Transkribering**: `services/transcription_service.py` + `components/transcription_monitor.py` – persistent `.cache/transcription_queue.json`, asyncio worker, `ui.timer` live-uppdatering
- **Nästa**: Fas 3 (nicegui_api_client + riktig backend-integration)

## Fas 3 Tasks
- [x] Skapa `services/nicegui_api_client.py` (httpx-wrapper mot er FastAPI)
- [x] Ersätt simulerad data med riktiga API-anrop (`/analyze_pipeline`, `/scan_process`)
- [x] Implementera polling-loop för transkriberingsstatus
- [x] Lägg till WebSocket-stöd för real-time logs/progress (valfritt i Fas 3)

### Fas 3 Status: ✅ Klar (2026-06-13, WebSocket 2026-06-13)
- **API-klient**: `app/nicegui_dashboard/services/nicegui_api_client.py` – `/health`, `/analyze_pipeline`, `/transcribe`, `/batch_transcribe`, `/scan_process`
- **Data**: Bakgrundsladdning från API vid start + manuell reload; fallback till demo-data
- **Live-analys**: `components/live_analysis.py` – pipeline via httpx med spinner och felhantering
- **Transkribering**: Riktiga API-anrop med strategival (transcribe/batch/scan_process), health-polling före batch
- **WebSocket**: `GET /ws/transcription` (backend) + `transcription_ws_client.py` (dashboard) – live loggar/progress via `X-Transcription-Job-Id`
- **Env**: `SENTIMENT_API_BASE_URL`, `SENTIMENT_API_KEY`, `SENTIMENT_API_TIMEOUT`
- **Nästa**: Fas 4 (polering, tester, deployment)

## Fas 4 Tasks
- [x] Teman & styling (ui.dark_mode, custom CSS)
- [x] Felhantering & loading states (ui.spinner, ui.notify)
- [x] Skriv tester (pytest + nicegui testing utilities)
- [x] Uppdatera README.md, ROADMAP.md och denna plan
- [x] Docker-compose för NiceGUI + backend

### Fas 4 Status: ✅ Klar (2026-06-13)
- **Tema**: `components/theme.py` – custom CSS, dark mode toggle i header
- **UX**: `services/ui_helpers.py` – centraliserade notify + `with_loading`
- **Tester**: `tests/test_nicegui_dashboard.py` – api client, demo provider, transcription, call detail
- **Deploy**: `docker-compose.nicegui.yml` – api + dashboard med healthcheck
- **Docs**: README.md utökad med NiceGUI quickstart
- **Nästa**: Fas 5 (avveckla Streamlit, feature flag)

## Fas 5 Tasks
- [x] Feature flag i main-appen
- [x] Parallell deployment
- [x] Användarutbildning / intern dokumentation
- [x] Ta bort Streamlit-kod

### Fas 5 Status: ✅ Klar (2026-06-13)
- **Feature flag**: `DASHBOARD_UI=nicegui` (default) via `app/dashboard_launcher.py` + `sentimentanalys-dashboard` CLI
- **Launcher**: `launcher/process_manager.py` startar NiceGUI (port 8080 default)
- **Borttaget**: `app/dashboard.py`, `app/nicegui_poc/`, `app/transcription_monitor.py`, `app/components/` (Streamlit)
- **Refaktorerat**: `data_services.py` / `demo_data.py` – `lru_cache` istället för `@st.cache_data`
- **Kvar**: `app/setup_hub.py` (Streamlit konfigurationshub, ej call center-dashboard)
- **Migrering klar** – NiceGUI är standarddashboard

## Fas 6.1: UX-förbättringar ✅ (2026-06-13)

### Uppgift 1 – Paginering + sökning i calls-tabellen ✅ (2026-06-13)
- [x] `services/calls_filter.py` – sök i call_id, title, agent, alla segment
- [x] `components/calls_table.py` – sökfält, paginering (10/20/50), föregående/nästa
- [x] `state.py` – `table_page`, `table_page_size`, `table_search`
- [x] Radklick → Call Detail bevarad

### Uppgift 2 – WebSocket reconnect ✅ (2026-06-13)
- [x] `transcription_ws_client.py` – exponential backoff, obegränsade försök under batch
- [x] UI-status: Connected / Reconnecting / Disconnected
- [x] Polling-fallback (2s) när WS nere under API-batch
- [x] Manuell **Reconnect WS**-knapp
- [x] Fix: `_start_ws_listener` startar nu korrekt (krävde inte job_id före skapande)

### Uppgift 3 – Plotly trajectory & agent trends ✅ (2026-06-13)
- [x] Ny flik **Analys & Trender** (`components/analytics_trends.py`)
- [x] `services/chart_data.py` – trajectory, agent empathy/QA, hot topics, escalation
- [x] `ui.plotly` interaktivt – klick på punkt/stapel → Call Detail
- [x] `plotly>=5.22.0` i `dashboard-nicegui` extra

### Uppgift 4 – Virtualisering av transkript ✅ (2026-06-13)
- [x] `services/transcript_virtualizer.py` – window/spacer-beräkning, sökfilter
- [x] `components/virtual_transcript.py` – `ui.scroll_area` + `@ui.refreshable` fönster
- [x] `call_detail.py` – virtualisering vid ≥40 segment, timeline + transkript
- [x] Klick i timeline scrollar transkript till rätt segment

### Uppgift 5 – Utökade tester ✅ (2026-06-13)
- [x] `tests/test_nicegui_dashboard.py` – API-klient, paginering, WS reconnect, TranscriptionState
- [x] `tests/test_nicegui_dashboard_ui.py` – NiceGUI User fixture rendering (overview, call detail, transcription, analytics)
- [x] `tests/fixtures/nicegui_test_pages.py` – isolerad test harness

## Fas 6.2: UX-polish (2026-06-13)

### Uppgift 1 – Global Reload i header ✅
- [x] `layout.py` – refresh-knapp i header (synlig när API-klient finns)
- [x] `main.py` – trådar `reload_from_api` till header; uppdaterar alla flikar

### Uppgift 2 – Tomma tillstånd ✅
- [x] `components/empty_state.py` – återanvändbar komponent
- [x] Översikt: ingen data, filter utan träffar, sök utan träffar
- [x] Transkribering: tom kö och tom logg

### Uppgift 3 – Sök-highlight i transkript ✅
- [x] `highlight_search_text()` i `transcript_virtualizer.py`
- [x] `<mark class="search-hit">` i `virtual_transcript.py`
- [x] Träffräknare under sökfält i Call Detail

### Uppgift 4 – Antal träffar ovanför tabellen ✅
- [x] `format_search_hit_label()` + prominent label i `calls_table.py`

### Uppgift 5 – Färgkodning QA-poäng ✅
- [x] `services/qa_display.py` – tier/CSS/chip-färger
- [x] Quasar-slot `body-cell-qa_score` i calls-tabellen
- [x] QA-chip i Call Detail

### Uppgift 6 – WebSocket reconnect
- Hoppad över (redan klar i Fas 6.1)

### Uppgift 7 – Plotly-tooltips ✅
- [x] `hovertemplate` för trajectory, agent trends, escalation i `chart_data.py`

### Uppgift 8 – Kollapsbar agent leaderboard ✅
- [x] `ui.expansion` hopfälld som standard i `overview.py`

### Uppgift 9 – Segment-räknare i timeline ✅
- [x] `[n/total]` på timeline-knappar + caption i `virtual_transcript.py`

# 4. Teknisk Mapping (Streamlit → NiceGUI)

| Streamlit-koncept          | NiceGUI-motsvarighet                  | Kommentar |
|----------------------------|---------------------------------------|---------|
| `st.metric` / `st.columns` | `ui.card` + labels, `ui.row`         | NiceGUI 3.x saknar `ui.metric` |
| `st.dataframe` / `st.table`| `ui.table`                           | Bra stöd |
| `st.expander`              | `ui.expansion`                       | NiceGUI 3.x namn |
| `st.button` + `on_click`   | `ui.button(on_click=...)`            | Event-driven (bättre) |
| `st.session_state`         | `app.storage` eller egna klasser     | Explicitare hantering |
| `st.rerun()`               | Inte behövs (reaktivitet inbyggd)    | Stor vinst |
| `st.status` / `st.progress`| `ui.status` eller custom progress    | Bra stöd |
| `st.tabs`                  | `ui.tabs` + `ui.tab_panels`          | Mycket bra |
| Plotly charts              | `ui.plotly` eller `ui.echart`        | Stöd finns |
| Bakgrundsjobb              | `asyncio.create_task` + `ui.timer`   | Mycket starkare |

# 5. State Management & Data Flow

- Använd `app.storage.user` eller egna Pydantic-modeller för session-state.
- För transkriberingskö: Fortsätt med JSON-persistens + utöka till Redis om skalning behövs.
- API-klient: Skapa en dedikerad `NiceGUIAPIClient` som wrappar httpx-anrop till er backend.

# 6. Risker & Mitigation

- **Risk**: Komplexa vyer blir svåra att port a
  **Mitigation**: Börja med PoC och iterera vy för vy.
- **Risk**: Prestanda på stora transkript
  **Mitigation**: Virtualisering + lazy loading i NiceGUI.
- **Risk**: Teamets inlärningskurva
  **Mitigation**: Bra intern dokumentation + pair-programming.

# 7. Acceptanskriterier per Fas

**Fas 1**: PoC körs och visar KPI + basic navigation utan krascher.
**Fas 2**: Alla tre huvudvyer fungerar interaktivt och matchar Streamlit-funktionalitet.
**Fas 3**: Riktiga backend-anrop fungerar + polling är stabil.
**Fas 4**: Appen är polerad, testad och dokumenterad.
**Fas 5**: Streamlit är avvecklad utan dataförlust.

# 8. Resurser & Verktyg

- NiceGUI docs: https://nicegui.io/
- Befintlig PoC: `app/nicegui_poc/main.py`
- API-dokumentation: `docs/API.md`
- Befintlig Streamlit-kod: `app/dashboard.py` + `app/components/`

# 9. Nästa steg efter denna plan

1. Godkänn planen internt.
2. Använd den optimala Grok Build-prompten nedan för att starta implementeringen.
3. Börja med Fas 1.

---

**Denna plan sparas i Git och är den officiella referensen för migreringen.**
