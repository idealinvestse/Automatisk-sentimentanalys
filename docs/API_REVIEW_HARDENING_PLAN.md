# Detaljerad Plan för API-genomgång, Tester, Förbättringar och Validering (Version 2.0)

**Projekt:** idealinvestse/Automatisk-sentimentanalys  
**Fokus:** REST API-lagret (src/api/) inklusive alla routers, schemas, app.py, integration med pipeline och Fas 4-funktioner  
**Datum:** 2026-06-03  
**Version:** 2.0 (uppdaterad och förbättrad efter initial review och live-fix)  
**Status:** Levande dokument – uppdateras iterativt under arbetet  
**Branch:** `api-review-v2-hardening`  
**Fas-status:** Fas 0 ✅ | Fas 1 ✅ | Fas 2 🔄 | Fas 3–5 ⏳  
**Referens:** Bygger på UTVECKLINGSPLAN.md (Fas 4 alla tasks DONE), tidigare REVIEW_MISTRAL_FAS3.md och den initiala API-reviewen 2026-06-03.

---

## 1. Sammanfattning och Reflektion över Tidigare Arbete

Den initiala planen var solid i struktur men kunde förbättras på följande punkter:
- Mer granulära tasks med explicita filändringar och kodexempel.
- Starkare kvantitativa framgångskriterier (coverage, latency, zero critical bugs).
- Bättre integration med befintlig kodbas (helpers.py, batch.py, core/errors.py, caching.py).
- Tydligare strategi för composer 2.5 + subagents (spawn, roller, iterationsloop).
- Riskhantering och backward-compat explicit.
- Live-fix av en bugg (deep_analysis copy-paste i pipeline.py) har redan utförts som exempel på iterativ process.

**Nuläge efter initial review + fix:**
- API har god grundstruktur (FastAPI + Pydantic v2 + custom exception handlers).
- Nya Fas 4 endpoints finns men innehåller inkonsistenser, ineffektiv kod och säkerhetsluckor.
- Tester finns men täckning och edge cases för Fas 4 är ofullständiga.
- En kritisk copy-paste-bugg fixad i commit 18af5c795fd1946b41dcb3c0f18f08d55af4ad5f.

**Mål med v2.0-planen:**
Skapa en production-ready, vältestad, säker och väldokumenterad API som fullt och korrekt exponerar hela stacken (Fas 1–4) utan dolda fel. Planen är optimerad för att köras med maximal användning av grok build / composer 2.5 subagents.

---

## 2. Framgångskriterier (Kvantitativa & Kvalitativa)

- **Testtäckning:** ≥ 90 % på src/api/ (mätt med pytest-cov).
- **Zero critical bugs:** Inga unhandled exceptions, schema-mismatch, eller säkerhetsluckor i produktionsläge.
- **Prestanda:** p99-latens för /analyze_pipeline < 2 s (med mockade tunga komponenter) eller acceptabel med caching.
- **Säkerhet:** Alla endpoints skyddade eller explicit markerade; llm_api_key hanteras säkert; inga secrets i logs/responses.
- **Dokumentation:** Full OpenAPI med exempel + dedikerad docs/API.md.
- **Backward compat:** Inga breaking changes utan explicit deprecation-period (minst en minor version).
- **Fas 4-validering:** Alla nya endpoints returnerar korrekta fält (agent_assessment, qa/compliance_qa, alerts, cached, etc.) och integrerar med pipeline + caching.

---

## 3. Detaljerad Fasindelad Plan

### Fas 0: Setup & Baseline (0,5 dag) — **DONE 2026-06-03**
**Tasks:**
1. ✅ Skapa branch `api-review-v2-hardening` från main.
2. ✅ `ruff check --fix src/api/` (1 E402 kvar → fixad i Fas 1); `mypy src/api/` (2 errors); `bandit` ej i venv → Fas 5.
3. ✅ `reports/api_openapi_baseline.json`
4. ✅ `pytest tests/test_api.py --cov=src/api` → **74.96%**, 14 passed, HTML: `reports/coverage_html`
5. ✅ Endpoint-inventory → `docs/API_FINDINGS.md`

**Baseline:** coverage 74.96%, ruff 0 errors (efter import-fix), mypy 2 errors, bandit pending.

### Fas 1: Djupgående Kodgranskning & Bugghunt (1 dag) — **DONE 2026-06-03**
**Fokus:** Statisk + semantisk analys av alla API-filer.  
**Findings:** `docs/API_FINDINGS.md` (P0×7, P1×12, P2×6)  
**Fix:** P0-1 `deep_analysis` på `/agent_performance` + regressionstest

**Tasks:**
1. Granska `src/api/app.py`: Exception handlers, lifespan, middleware-frånvaro, CORS, tracing.
2. Granska `src/api/schemas.py`: Alla Fas 4 request/response-modeller – lägg till saknade fält (deep_analysis i AgentPerformanceRequest m.fl.), striktare Field-valideringar, examples.
3. Granska varje router:
   - `pipeline.py` (redan delvis fixad): Ta bort getattr-hacks, inför DI, lägg till storleksbegränsningar.
   - `transcription.py`, `conversation.py`, `text.py`, `scan.py`, `batch.py`: Samma mönster.
4. Identifiera alla "code smells": breda except, duplicerad pipeline-instansiering, inkonsekvent llm_api_key-hantering, saknad rate limiting, svag error taxonomy.
5. Skapa `API_FINDINGS.md` (eller uppdatera denna fil) med kategoriserade issues (P0/P1/P2).

**Exempel på redan fixad P0-bugg (live):** 
I `pipeline.py` hade `deep_analysis=req.use_mistral_llm` i flera endpoints – nu korrigerat till explicit + getattr-fallback. Detta är modell för hur iterativ fix ska se ut.

**Deliverables:** Uppdaterad findings-lista, förslag på refaktoriseringar.

### Fas 2: Förbättringar & Refaktorering (1,5–2 dagar) — **IN PROGRESS**
**Klart (2026-06-03):** `dependencies.py`, `settings.py`, shared cache/alert lifespan, API key auth (opt-in via `SENTIMENT_API_KEY`), CORS env, request-ID + security headers, `LLMError` handler, Fas4 `deep_analysis` på alla requests, payload limits, korrekt `cache_hit`, pipeline DI, version 0.4.0, coverage ~78%.  
**Kvar:** rate limiting, RFC 7807, `API_MEDIA_ROOT` sandbox, övriga routers exception taxonomy, rate limit, ≥90% coverage (Fas 3).

**Prioriterad ordning (högst ROI först):**

**2.1 Säkerhet & Auth (hög prioritet)**
- Inför `APIKeyHeader` eller `OAuth2PasswordBearer` via `Depends`.
- Maskera llm_api_key i alla logs.
- Lägg till rate limiting (slowapi eller custom middleware).
- Lägg till CORS-middleware med env-konfiguration.
- Security headers middleware.
- Uppdatera exception handlers till RFC 7807 Problem Details format.

**2.2 Dependency Injection & Testbarhet**
- Skapa `src/api/dependencies.py` med:
  - `get_pipeline(profile: str | None = None, ...)` → CallAnalysisPipeline
  - `get_cache()` → AggregateCache
  - `get_alert_engine()`
- Refaktorera alla routers att använda `Depends`.

**2.3 Prestanda & Caching**
- Se till att alla Fas 4 endpoints använder `pipe.get_cached_*` där möjligt.
- Lägg till `max_segments: int = Field(..., le=200)` i relevanta request-modeller.
- Överväg BackgroundTasks för tunga anrop eller introducera enkel task queue.

**2.4 Schemas & Validering**
- Lägg till `deep_analysis: bool = Field(False)` i alla Fas 4 request-klasser.
- Lägg till `Field(..., examples=[...])` och `json_schema_extra` för bättre OpenAPI.
- Striktare validators (t.ex. agent_id regex, segment text length).

**2.5 Övriga förbättringar**
- Bump API-version till "0.4.0" eller "1.0.0" i app.py.
- Förbättra `helpers.py` och `batch.py` om de används av routers.
- Lägg till request ID tracing (middleware + log).

**Deliverables:** Refaktorerad kod, nya dependencies.py, uppdaterade schemas.

### Fas 3: Testutveckling & Validering (1,5 dagar)
**Tasks:**
1. Analysera befintlig `tests/test_api.py` – identifiera gap för Fas 4 endpoints.
2. Skapa/utöka tester:
   - Unit tests för varje ny endpoint (mock pipeline & cache).
   - Integration tests med TestClient + sample segments.
   - Error path tests (LLM fail, invalid input, cache miss, large payload).
   - Property-based tests med hypothesis på schemas.
   - Contract tests: verifiera att response alltid innehåller förväntade Fas 4-fält.
3. Kör full suite iterativt efter varje fix: `pytest ... --cov --cov-fail-under=90`.
4. Lägg till pre-commit hooks eller CI-steg för API (om inte finns).

**Deliverables:** ≥ 90 % coverage, alla tester gröna, nya testfiler eller utökningar.

### Fas 4: Dokumentation & OpenAPI (0,5 dag)
**Tasks:**
1. Berika alla endpoints med detaljerade docstrings, tags, summaries, descriptions.
2. Lägg till rika examples i Pydantic-modeller.
3. Skapa/uppdatera `docs/API.md` med:
   - Quickstart
   - Exempel för varje endpoint (curl + Python)
   - Autentisering & rate limiting
   - Error codes
   - Fas 4-specifika användningsfall
4. Uppdatera README.md med länk till docs/API.md.

**Deliverables:** Full OpenAPI + dedikerad dokumentation.

### Fas 5: Slutvalidering, CI & Leverans (0,5 dag)
**Tasks:**
1. Manuell smoke test via uvicorn + /docs.
2. Validera mot alla acceptance criteria i UTVECKLINGSPLAN.md Fas 4.
3. Kör load-test stub (t.ex. med locust eller enkelt script) på nyckel-endpoints.
4. Uppdatera denna plan med status per task.
5. Merge till main efter godkännande.
6. Skapa release note / changelog-entry.

**Deliverables:** Sign-off, uppdaterad plan, merge-ready kod.

---

## 4. Riskhantering

| Risk                        | Sannolikhet | Påverkan | Mitigation                                      |
|-----------------------------|-------------|----------|-------------------------------------------------|
| Breaking changes            | Låg         | Hög      | Använd deprecation + version i URL/header       |
| LLM-kostnader via API       | Medel       | Medel    | Rate limit + caching + explicit --use_mistral_llm default False |
| Prestanda på stora batcher  | Medel       | Hög      | Max limits + caching + async/background         |
| Komplexitet i DI-refaktor   | Låg         | Medel    | Börja med pipeline-dependency, iterera          |
| Otillräcklig test coverage  | Medel       | Hög      | Tvingande cov-fail-under + subagent TestAgent   |

---

## 5. Hur man Använder Denna Plan med Composer 2.5 / Grok Build

1. Klistra in den "Ultimata prompten" (se separat avsnitt nedan).
2. Composer/spawnade subagents måste läsa denna fil (`docs/API_REVIEW_HARDENING_PLAN.md`) som primär källa.
3. Följ faserna strikt i ordning (0 → 5).
4. Varje fas avslutas med commit + uppdatering av status i denna fil.
5. Använd spawn subagent för parallellt arbete (t.ex. en ImplementationAgent per router + en TestAgent).
6. Iterativ loop: Fix → Test → Validate → Document → Commit.
7. Vid osäkerhet: anta production best practice och dokumentera valet i findings.

---

## 6. Bilaga: Redan Fixade & Prioriterade Issues (exempel)

**Redan fixade (live 2026-06-03):**
- `/analyze_pipeline`: `deep_analysis=req.deep_analysis` korrekt.
- **P0-1 (Fas 1):** `/agent_performance` — `deep_analysis` var felaktigt `req.use_mistral_llm`; korrigerat + schema + regressionstest.

**Högprioriterade att fixa i Fas 1–2:**
- Saknade `deep_analysis` fält i Fas 4 request schemas.
- Breda `except Exception` → använd specifika domain errors.
- Inkonsekvent llm_api_key-hantering.
- Avsaknad av DI, rate limiting, CORS, tracing.
- Låg testtäckning för nya endpoints.
- Ineffektiv pipeline-instansiering i varje request.

---

**Slut på plan v2.0**  
Denna plan är nu den officiella, uppdaterade och förbättrade versionen. Alla framtida arbete på API:n ska referera till och följa denna plan.