# GROK BUILD PLAN – Fas 1–3

**Projekt:** Automatisk-sentimentanalys  
**Syfte:** Validering, Dokumentation & Dashboard-visualisering av Fas 4  
**Version:** v0.4.x → v0.5-prep  
**Tidsestimat:** 3–5 veckor  
**Senast uppdaterad:** 2026-06-19

Denna plan är optimerad för **Grok Build** och andra coding agents. Följ den stegvis.

## Övergripande instruktioner för agenten

- **Iterativt arbetssätt**: Implementera → Testa → Dokumentera → Commit
- **Följ projektkonventioner**: Pydantic, EvidenceSpan, hybrid local+LLM, DashboardState, NiceGUIAPIClient, refresh-callbacks
- **Kvalitet**: Non-fatal error handling, logging, tester, uppdaterad dokumentation
- **Scope**: Håll dig till planen. Notera eventuella out-of-scope-uppgifter separat.

---

## Fas 1: Omedelbar validering & kvalitetssäkring
**Prioritet:** Hög  
**Estimat:** 1–2 veckor  
**Mål:** Bekräfta att Fas 4 är stabil och produktionsredo

### Delsteg

1. **Full testsvit & coverage**
   - Kör: `pytest tests/ -q --cov=src --cov-report=term-missing --cov-fail-under=85`
   - Fokusera på API-, pipeline- och evaluate-tester samt nya Fas 4-moduler.
   - Fixa failing tests och coverage-gaps.

2. **Utvärderingskörning**
   - Kör `python -m src.evaluate` (alla lägen inkl. llm-quality).
   - Verifiera Fas 4-metrics (agent_performance, qa, insights, alerts).
   - Skapa rapport: `reports/evaluate_fas4_validation.md`

3. **End-to-end smoke-tester**
   - Testa `analyze-call` med `--use-mistral-llm` på samples.
   - Testa batch via API (`/scan_process`, `/analyze_pipeline`).
   - Verifiera caching, pre-computation och PII-redaction + LLM-anrop.

4. **Prestanda & edge cases**
   - Testa långa ljudfiler och låg confidence-scenario.
   - Verifiera alla nya endpoints (`/agent_performance/{id}`, `/insights/hot_topics`, `/qa/score`, `/search/semantic`, `/alerts`).

### Acceptanskriterier
- Alla tester gröna med ≥ 85–90 % coverage på relevanta moduler.
- `evaluate` visar meningsfulla Fas 4-resultat.
- Inga krascher vid PII-redaction + LLM.
- Caching och pre-computation fungerar korrekt.

**Påverkade filer:** `tests/`, `src/evaluate.py`, `reports/`, eventuella fixes i `pipeline.py` och Fas 4-moduler.

---

## Fas 2: Dokumentation & Release
**Prioritet:** Hög  
**Estimat:** 3–5 dagar  
**Mål:** Göra projektet release-ready

### Delsteg

1. **CHANGELOG.md**
   - Lägg till sektion för v0.4.1 / v0.5-prep.
   - Sammanfatta alla Fas 4-leveranser.

2. **README.md & Quickstart**
   - Uppdatera "Funktioner (v0.4+)" med nya Fas 4-kapaciteter.
   - Lägg till exempel på agent performance, QA-score och hot topics via CLI/API/dashboard.

3. **docs/API.md**
   - Dokumentera de 5 nya endpoints med request/response-exempel och curl.
   - Inkludera autentisering och rate limit.

4. **ROADMAP.md & UTVECKLINGSPLAN.md**
   - Markera Fas 4 som **Slutförd**.
   - Uppdatera "Nästa steg" till Fas 2 (data/finetuning) + Dashboard-visualisering + produktion.

5. **Övrigt**
   - Synka `docs/ARCHITECTURE.md` vid behov.
   - Skapa `docs/FAS4_COMPLETION.md` (valfritt men rekommenderat).

### Acceptanskriterier
- CHANGELOG och README är uppdaterade och korrekta.
- Alla nya API-endpoints är väldokumenterade med exempel.
- En extern person/agent kan använda de nya funktionerna enbart via dokumentationen.

**Påverkade filer:** `CHANGELOG.md`, `README.md`, `docs/API.md`, `ROADMAP.md`, `UTVECKLINGSPLAN.md`

---

## Fas 3: Dashboard & Användarupplevelse – NiceGUI
**Prioritet:** Hög–Medel  
**Estimat:** 2–4 veckor  
**Mål:** Synliggöra Fas 4-funktionerna i UI

### Befintlig struktur att följa
- Använd `DashboardState`
- Skapa komponenter i `app/nicegui_dashboard/components/`
- Lägg till ny `ui.tab` + `ui.tab_panel` i `main.py`
- Använd refresh-callback-mönster
- Hämta data via `NiceGUIAPIClient`

### Delsteg (rekommenderad ordning)

1. **Agent Performance-vy**
   - Ny flik/sektion "Agent Performance"
   - Visa per-agent metrics (empathy, talk ratio, de-escalation, compliance flags)
   - Trend-graf (Plotly) + leaderboard
   - Drill-down till specifikt samtal

2. **QA & Compliance-vy**
   - "QA Scorecard" sektion
   - Visa overall QA-score, passed/failed criteria, risk level
   - Möjlighet att välja scorecard och se detaljerad evidence

3. **Insights & Hot Topics**
   - Utöka eller ny flik "Insikter & Trender"
   - Lista hot topics med volym, sentiment, trend
   - Root cause clusters + actionable recommendations
   - Integration med semantic search

4. **Alerts & Actions**
   - "Alerts" panel eller badge i header/översikt
   - Lista aktiva alerts med severity + recommended actions
   - Möjlighet att markera som hanterad (stub)

5. **Semantic Search i UI**
   - Sökfält + filter
   - Ranked resultat med highlights och evidence spans
   - Klicka för att öppna i Call Detail

### Tekniska riktlinjer
- Återanvänd befintliga `EvidenceSpan`, Pydantic-modeller och `load_from_api`-mönster.
- Skapa återanvändbara komponenter (t.ex. `render_agent_metrics_table`).
- Lägg till loading states och `ui.notify` error handling.
- Stöd både demo-data och riktig API-data.

### Acceptanskriterier
- Alla nya vyer är synliga och fungerar.
- Data hämtas korrekt från API.
- Användaren kan navigera från översikt → agent metrics → specifikt samtal.
- UI följer befintlig dark theme och är responsivt.

**Påverkade filer (exempel):**
- `app/nicegui_dashboard/main.py`
- `app/nicegui_dashboard/components/` (nya filer: `agent_performance.py`, `qa_scorecard.py`, `insights_hot_topics.py`, `alerts_panel.py` m.fl.)
- `app/nicegui_dashboard/state.py` (vid behov)
- `app/nicegui_dashboard/services/nicegui_api_client.py` (vid behov)

---

## Sammanfattning & Rekommenderad körordning

| Fas | Fokus                          | Tid     | Beroenden     | Kommentar                     |
|-----|--------------------------------|---------|---------------|-------------------------------|
| 1   | Validering & Tester            | 1–2 v   | -             | Börja här                     |
| 2   | Dokumentation & Release        | 3–5 d   | Delvis Fas 1  | Gör det "officiellt"        |
| 3   | Dashboard-visualisering        | 2–4 v   | Fas 1 + 2     | Kan delvis startas tidigt     |

**Totalt estimat:** 3–5 veckor vid dedikerad insats.

---

## Hur du använder denna plan i Grok Build

1. Kopiera denna fil till `docs/GROK_BUILD_PLAN_FAS1-3.md` i repot.
2. Börja med **Fas 1**.
3. Efter varje större del: commit + push.
4. Använd den optimala prompten nedan för att låta Grok exekvera planen.

**Optimal prompt att klistra in i Grok Build:**

```
Du är en expert coding agent som arbetar på projektet https://github.com/idealinvestse/Automatisk-sentimentanalys.

Läs in den senaste versionen av filen `docs/GROK_BUILD_PLAN_FAS1-3.md` från main-branchen.

Följ planen **exakt** steg för steg. Börja med Fas 1.

Arbeta iterativt:
- Implementera en del
- Testa lokalt (pytest, evaluate, smoke-tester)
- Uppdatera dokumentation
- Commit med tydligt meddelande
- Push till main

Använd de befintliga verktygen för att läsa filer, köra kommandon och göra ändringar i repot.

När du är klar med en Fas, bekräfta och gå vidare till nästa.

Håll dig strikt till scopet i planen.
```