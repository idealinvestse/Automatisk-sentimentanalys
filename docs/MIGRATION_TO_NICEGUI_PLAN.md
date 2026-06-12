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
- [ ] Skapa `app/nicegui_dashboard/` struktur (main.py, components/, services/)
- [ ] Grundläggande layout med ui.header, ui.tabs, ui.tab_panels
- [ ] Portning av KPI-metrics (ui.metric eller ui.card)
- [ ] Enkel tabell-komponent (ui.table)
- [ ] Validering att PoC matchar Streamlit-funktionalitet

## Fas 2 Tasks
- [ ] Översiktsvy: Hot Topics chips, Agent Leaderboard tabell, filtrerad calls-tabell med on_click
- [ ] Call Detail: ui.card för header, timeline (ui.timeline eller custom), sökbart transcript (ui.textarea + markdown)
- [ ] Structured Insights: ui.expander + ui.markdown för LLM + Fas4-data
- [ ] Transkriberings Monitor: Full implementation av persistent JSON-kö, start/paus/stopp, live progress, loggruta, settings-form

## Fas 3 Tasks
- [ ] Skapa `services/nicegui_api_client.py` (httpx-wrapper mot er FastAPI)
- [ ] Ersätt simulerad data med riktiga API-anrop (`/analyze_pipeline`, `/scan_process`)
- [ ] Implementera polling-loop för transkriberingsstatus
- [ ] Lägg till WebSocket-stöd för real-time logs/progress (valfritt i Fas 3)

## Fas 4 Tasks
- [ ] Teman & styling (ui.dark_mode, custom CSS)
- [ ] Felhantering & loading states (ui.spinner, ui.notify)
- [ ] Skriv tester (pytest + nicegui testing utilities)
- [ ] Uppdatera README.md, ROADMAP.md och denna plan
- [ ] Docker-compose för NiceGUI + backend

## Fas 5 Tasks
- [ ] Feature flag i main-appen
- [ ] Parallell deployment
- [ ] Användarutbildning / intern dokumentation
- [ ] Ta bort Streamlit-kod

# 4. Teknisk Mapping (Streamlit → NiceGUI)

| Streamlit-koncept          | NiceGUI-motsvarighet                  | Kommentar |
|----------------------------|---------------------------------------|---------|
| `st.metric` / `st.columns` | `ui.metric`, `ui.row`, `ui.card`     | Mycket likt |
| `st.dataframe` / `st.table`| `ui.table`                           | Bra stöd |
| `st.expander`              | `ui.expander`                        | Direkt motsvarighet |
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
