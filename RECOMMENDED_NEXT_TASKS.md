# RECOMMENDED NEXT TASKS — Automatisk-sentimentanalys

**Generated:** 2026-06-26 via github-repo-deep-dive skill  
**After clarifying questions with user**  
**Based on fresh PROJECT_STATUS.md + AGENT_CONTEXT.md (2026-06-26)**

## Understanding & Rationale

Användaren har tydligt prioriterat:
- **Huvudmål närmaste 2–4 veckor**: En **polerad Fas 4-dashboard som QA-team faktiskt använder dagligen**.
- Starkt fokus på **LLM-judge-integration + Dashboard-visualisering/UX** samt Alerting/webhook.
- Föredrar **foundational improvements** som gör allt framtida arbete mycket enklare (istället för rena quick wins).
- Långsiktigt: Bättre UX för call center-personal + högre reliability.

**Deep dive-insikt**: `src/analysis/llm_judge.py` är redan väl implementerat (batchning, budget-guard, graceful fallback, EXTERNAL LLM CALL-logging, Pydantic-schemas). Problemet är att den **inte är wired** in i `CallAnalysisPipeline`, inte exponerad via API och inte visualiserad i NiceGUI-dashboard. Detta är den största "missing link" just nu.

Därför blir de högst rankade uppgifterna de som:
1. Gör LLM-judge till en förstklassig, pålitlig del av pipelinen (foundational).
2. Ger synlig, användbar visualisering i dashboarden (direkt kopplat till huvudmålet).
3. Förbättrar reliability och dataflöde så att dashboarden känns snabb och stabil för daglig användning.

## Ranked Task List

### TASK-01: Wire LLMJudgeAnalyzer fullt in i CallAnalysisPipeline + API
**Why this task now**: Högst värde + foundational. LLM-judge finns implementerat men används inte i det riktiga flödet. När den är wired kan dashboarden konsumera verdict pålitligt och QA-team får direkt nytta på låg-confidence segment (där det behövs mest). Bygger direkt på befintlig kod i `src/analysis/llm_judge.py`.

**Description**:
- Lägg till `llm_judge` i `CallAnalysisPipeline` (efter `sentiment` i topo-sort).
- Exponera resultatet i API-schemas och `/pipeline` + `/analyze` endpoints.
- Uppdatera `ctx.results` och `report.results` så att downstream (dashboard, caching, alerting) kan använda det.
- Lägg till grundläggande tester i `tests/test_pipeline.py` och `tests/test_llm_judge.py`.

**Primary files / components**:
- `src/pipeline.py` (main pipeline wiring)
- `src/api/schemas.py` + `src/api/routers/pipeline.py`
- `src/analysis/llm_judge.py` (små justeringar om behövs)
- `tests/test_pipeline.py` + `tests/test_api.py`

**Estimated effort**: Medium (1–2 sessioner) — befintlig implementation är ren och väldokumenterad.

**Dependencies / prerequisites**: Fresh `AGENT_CONTEXT.md` + `PROJECT_STATUS.md` (har vi). Inga andra tasks.

**Expected impact / value**: Mycket hög. Gör att Fas 4-dashboard kan visa actionable judge-verdicts. Unblockar TASK-02.

**Risks / things to watch**: Budget/cost tracking i judge måste respekteras. Fallback-beteende måste vara graceful.

**Success criteria**:
- `llm_judge` körs automatiskt på låg-confidence segment i pipeline.
- Resultat syns i API-response och sparas i report.
- Inga regressioner i befintliga sentiment/intent-flöden (alla tester gröna).

---

### TASK-02: Skapa/utöka NiceGUI-komponent för LLM-judge verdicts + integrera i Fas 4-dashboard
**Why this task now**: Direkt kopplat till användarens huvudmål ("polerad Fas 4-dashboard som QA-team använder dagligen"). Ger synlig nytta och gör dashboarden mer komplett.

**Description**:
- Skapa ny komponent `llm_judge_panel.py` eller utöka befintlig (t.ex. `fas4_insights.py` eller `call_detail.py`).
- Visa per-segment: original sentiment + judge_label + judge_confidence + reasoning (på svenska).
- Integrera i `call_detail` flik och/eller ny "LLM Judge Insights" sektion.
- Använd `NiceGUIAPIClient` för att hämta data (eller local `fas4_data.py`).
- Lägg till enkel filtering/sortering på low-confidence segments.

**Primary files / components**:
- `app/nicegui_dashboard/components/` (ny fil eller utökning)
- `app/nicegui_dashboard/services/nicegui_api_client.py` + `fas4_data.py`
- `app/nicegui_dashboard/main.py` eller layout (för att lägga till flik/sektion)

**Estimated effort**: Medium (1–2 sessioner) — NiceGUI-komponenter är väletablerade i projektet.

**Dependencies / prerequisites**: TASK-01 (pipeline + API måste leverera judge-data).

**Expected impact / value**: Hög. Gör dashboarden mer användbar för QA-team direkt.

**Risks / things to watch**: Håll UI ren och inte överbelastad. Använd befintliga mönster från `emotion_timeline.py` och `insights_hot_topics.py`.

**Success criteria**:
- LLM-judge verdicts visas snyggt och begripligt i dashboarden.
- QA-användare kan se varför ett segment fick en viss bedömning.
- Inga prestandaproblem (använd caching).

---

### TASK-03: Förbättra caching + async data fetching i NiceGUI dashboard (foundational reliability)
**Why this task now**: Foundational improvement som gör dashboarden snabbare och mer pålitlig för daglig användning. Stödjer både TASK-02 och långsiktigt UX/reliability-mål.

**Description**:
- Se över `NiceGUIAPIClient` och `fas4_data.py` — se till att alla Fas 4-endpoints använder cache på rätt sätt.
- Gör async hämtning utanför `@ui.refreshable` där det behövs (följ mönster från tidigare fixes).
- Lägg till loading states och error handling i komponenter.
- Överväg att utöka `AggregateCache` för dashboard-specifika queries.

**Primary files / components**:
- `app/nicegui_dashboard/services/nicegui_api_client.py`
- `app/nicegui_dashboard/services/fas4_data.py`
- `src/caching.py`
- Flera dashboard-komponenter (för loading/error states)

**Estimated effort**: Medium — mycket av grundarbetet finns redan.

**Dependencies / prerequisites**: Inga hårda, men synergi med TASK-01/02.

**Expected impact / value**: Hög foundational. Dashboarden känns proffsigt snabb och stabil.

**Risks / things to watch**: Cache invalidation måste vara korrekt.

**Success criteria**:
- Dashboard laddar snabbt även med många samtal.
- Inga stale data-problem.
- Bra UX vid nätverksproblem eller långsamma backend-anrop.

---

### TASK-04: Polera Alerting webhook till produktionskvalitet + enkel UI i test_lab
**Why this task now**: En av de delar användaren nämnde som smärtsam. Webhook har redan circuit breaker och retries, men saknar enkel testbarhet och mark-as-handled-flöde.

**Description**:
- Lägg till "Test Webhook" knapp i `test_lab.py`.
- Implementera "Mark as handled" för alerts i dashboard (stub → riktig).
- Förbättra error logging och retry-beteende om nödvändigt.
- Uppdatera `configs/alerting_config.yaml` dokumentation.

**Primary files / components**:
- `app/nicegui_dashboard/components/test_lab.py`
- `app/nicegui_dashboard/components/alerts_panel.py`
- `src/alerting.py`
- `configs/alerting_config.yaml`

**Estimated effort**: Small–Medium.

**Dependencies / prerequisites**: Inga hårda.

**Expected impact / value**: Medel — gör alerting mer användbart i praktiken.

**Risks / things to watch**: Håll det enkelt; det är inte huvudfokus just nu.

**Success criteria**:
- Webhook kan testas enkelt från dashboard.
- Alerts kan markeras som hanterade.

---

## How to use this file
En agent (eller du) bör:
1. Välja en task (börja med TASK-01 eller TASK-02 beroende på vad som känns mest prioriterat).
2. Läsa relevanta delar av `AGENT_CONTEXT.md` + `PROJECT_STATUS.md`.
3. Implementera.
4. Efter avslut: Kör `github-project-status` igen (och eventuellt denna deep-dive skill).

**Rekommenderad ordning just nu**: TASK-01 → TASK-02 → TASK-03 (foundational + direkt nytta för dashboard).