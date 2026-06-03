# Detaljerad Plan f├Âr API-genomg├Ñng, Tester, F├Ârb├ñttringar och Validering (Version 2.0)

**Projekt:** idealinvestse/Automatisk-sentimentanalys  
**Fokus:** REST API-lagret (src/api/) inklusive alla routers, schemas, app.py, integration med pipeline och Fas 4-funktioner  
**Datum:** 2026-06-03  
**Version:** 2.0 (uppdaterad och f├Ârb├ñttrad efter initial review och live-fix)  
**Status:** Levande dokument ÔÇô uppdateras iterativt under arbetet  
**Branch:** `api-review-v2-hardening`  
**Fas-status:** Fas 0ÔÇô5 Ô£à **SIGN-OFF 2026-06-03** ÔÇö branch `api-review-v2-hardening` merge-ready  
**Referens:** Bygger p├Ñ UTVECKLINGSPLAN.md (Fas 4 alla tasks DONE), tidigare REVIEW_MISTRAL_FAS3.md och den initiala API-reviewen 2026-06-03.

---

## 1. Sammanfattning och Reflektion ├Âver Tidigare Arbete

Den initiala planen var solid i struktur men kunde f├Ârb├ñttras p├Ñ f├Âljande punkter:
- Mer granul├ñra tasks med explicita fil├ñndringar och kodexempel.
- Starkare kvantitativa framg├Ñngskriterier (coverage, latency, zero critical bugs).
- B├ñttre integration med befintlig kodbas (helpers.py, batch.py, core/errors.py, caching.py).
- Tydligare strategi f├Âr composer 2.5 + subagents (spawn, roller, iterationsloop).
- Riskhantering och backward-compat explicit.
- Live-fix av en bugg (deep_analysis copy-paste i pipeline.py) har redan utf├Ârts som exempel p├Ñ iterativ process.

**Nul├ñge efter initial review + fix:**
- API har god grundstruktur (FastAPI + Pydantic v2 + custom exception handlers).
- Nya Fas 4 endpoints finns men inneh├Ñller inkonsistenser, ineffektiv kod och s├ñkerhetsluckor.
- Tester finns men t├ñckning och edge cases f├Âr Fas 4 ├ñr ofullst├ñndiga.
- En kritisk copy-paste-bugg fixad i commit 18af5c795fd1946b41dcb3c0f18f08d55af4ad5f.

**M├Ñl med v2.0-planen:**
Skapa en production-ready, v├ñltestad, s├ñker och v├ñldokumenterad API som fullt och korrekt exponerar hela stacken (Fas 1ÔÇô4) utan dolda fel. Planen ├ñr optimerad f├Âr att k├Âras med maximal anv├ñndning av grok build / composer 2.5 subagents.

---

## 2. Framg├Ñngskriterier (Kvantitativa & Kvalitativa)

- **Testt├ñckning:** ÔëÑ 90 % p├Ñ src/api/ (m├ñtt med pytest-cov).
- **Zero critical bugs:** Inga unhandled exceptions, schema-mismatch, eller s├ñkerhetsluckor i produktionsl├ñge.
- **Prestanda:** p99-latens f├Âr /analyze_pipeline < 2 s (med mockade tunga komponenter) eller acceptabel med caching.
- **S├ñkerhet:** Alla endpoints skyddade eller explicit markerade; llm_api_key hanteras s├ñkert; inga secrets i logs/responses.
- **Dokumentation:** Full OpenAPI med exempel + dedikerad docs/API.md.
- **Backward compat:** Inga breaking changes utan explicit deprecation-period (minst en minor version).
- **Fas 4-validering:** Alla nya endpoints returnerar korrekta f├ñlt (agent_assessment, qa/compliance_qa, alerts, cached, etc.) och integrerar med pipeline + caching.

---

## 3. Detaljerad Fasindelad Plan

### Fas 0: Setup & Baseline (0,5 dag) ÔÇö **DONE 2026-06-03**
**Tasks:**
1. Ô£à Skapa branch `api-review-v2-hardening` fr├Ñn main.
2. Ô£à `ruff check --fix src/api/` (1 E402 kvar ÔåÆ fixad i Fas 1); `mypy src/api/` (2 errors); `bandit` ej i venv ÔåÆ Fas 5.
3. Ô£à `reports/api_openapi_baseline.json`
4. Ô£à `pytest tests/test_api.py --cov=src/api` ÔåÆ **74.96%**, 14 passed, HTML: `reports/coverage_html`
5. Ô£à Endpoint-inventory ÔåÆ `docs/API_FINDINGS.md`

**Baseline:** coverage 74.96%, ruff 0 errors (efter import-fix), mypy 2 errors, bandit pending.

### Fas 1: Djupg├Ñende Kodgranskning & Bugghunt (1 dag) ÔÇö **DONE 2026-06-03**
**Fokus:** Statisk + semantisk analys av alla API-filer.  
**Findings:** `docs/API_FINDINGS.md` (P0├ù7, P1├ù12, P2├ù6)  
**Fix:** P0-1 `deep_analysis` p├Ñ `/agent_performance` + regressionstest

**Tasks:**
1. Granska `src/api/app.py`: Exception handlers, lifespan, middleware-fr├Ñnvaro, CORS, tracing.
2. Granska `src/api/schemas.py`: Alla Fas 4 request/response-modeller ÔÇô l├ñgg till saknade f├ñlt (deep_analysis i AgentPerformanceRequest m.fl.), striktare Field-valideringar, examples.
3. Granska varje router:
   - `pipeline.py` (redan delvis fixad): Ta bort getattr-hacks, inf├Âr DI, l├ñgg till storleksbegr├ñnsningar.
   - `transcription.py`, `conversation.py`, `text.py`, `scan.py`, `batch.py`: Samma m├Ânster.
4. Identifiera alla "code smells": breda except, duplicerad pipeline-instansiering, inkonsekvent llm_api_key-hantering, saknad rate limiting, svag error taxonomy.
5. Skapa `API_FINDINGS.md` (eller uppdatera denna fil) med kategoriserade issues (P0/P1/P2).

**Exempel p├Ñ redan fixad P0-bugg (live):** 
I `pipeline.py` hade `deep_analysis=req.use_mistral_llm` i flera endpoints ÔÇô nu korrigerat till explicit + getattr-fallback. Detta ├ñr modell f├Âr hur iterativ fix ska se ut.

**Deliverables:** Uppdaterad findings-lista, f├Ârslag p├Ñ refaktoriseringar.

### Fas 2: F├Ârb├ñttringar & Refaktorering (1,5ÔÇô2 dagar) ÔÇö **DONE 2026-06-03**
**Levererat:** `dependencies.py`, `settings.py`, `path_validation.py`, `router_errors.py`, shared cache/alert, API key auth, CORS, request-ID + security headers, `LLMError` handler, Fas4 schemas, payload limits, `cache_hit`, DI, v0.4.0, saniterade fel p├Ñ alla routers, `API_MEDIA_ROOT` sandbox.  
**Kvar (l├Ñg prio):** rate limiting, RFC 7807 Problem Details.

**Prioriterad ordning (h├Âgst ROI f├Ârst):**

**2.1 S├ñkerhet & Auth (h├Âg prioritet)**
- Inf├Âr `APIKeyHeader` eller `OAuth2PasswordBearer` via `Depends`.
- Maskera llm_api_key i alla logs.
- L├ñgg till rate limiting (slowapi eller custom middleware).
- L├ñgg till CORS-middleware med env-konfiguration.
- Security headers middleware.
- Uppdatera exception handlers till RFC 7807 Problem Details format.

**2.2 Dependency Injection & Testbarhet**
- Skapa `src/api/dependencies.py` med:
  - `get_pipeline(profile: str | None = None, ...)` ÔåÆ CallAnalysisPipeline
  - `get_cache()` ÔåÆ AggregateCache
  - `get_alert_engine()`
- Refaktorera alla routers att anv├ñnda `Depends`.

**2.3 Prestanda & Caching**
- Se till att alla Fas 4 endpoints anv├ñnder `pipe.get_cached_*` d├ñr m├Âjligt.
- L├ñgg till `max_segments: int = Field(..., le=200)` i relevanta request-modeller.
- ├ûverv├ñg BackgroundTasks f├Âr tunga anrop eller introducera enkel task queue.

**2.4 Schemas & Validering**
- L├ñgg till `deep_analysis: bool = Field(False)` i alla Fas 4 request-klasser.
- L├ñgg till `Field(..., examples=[...])` och `json_schema_extra` f├Âr b├ñttre OpenAPI.
- Striktare validators (t.ex. agent_id regex, segment text length).

**2.5 ├ûvriga f├Ârb├ñttringar**
- Bump API-version till "0.4.0" eller "1.0.0" i app.py.
- F├Ârb├ñttra `helpers.py` och `batch.py` om de anv├ñnds av routers.
- L├ñgg till request ID tracing (middleware + log).

**Deliverables:** Refaktorerad kod, nya dependencies.py, uppdaterade schemas.

### Fas 3: Testutveckling & Validering (1,5 dagar) ÔÇö **DONE 2026-06-03**
**Resultat:** `tests/test_api_coverage.py` (+34 tester), **52 passed**, **`src/api` coverage 96.64%** (m├Ñl ÔëÑ90%).  
**K├Âr:** `pytest tests/test_api.py tests/test_api_coverage.py --cov=src/api --cov-fail-under=90`

**Tasks:**
1. Ô£à Analysera befintlig `tests/test_api.py` ÔÇô identifiera gap f├Âr Fas 4 endpoints.
2. Ô£à Skapa/ut├Âka tester:
   - Unit tests f├Âr varje ny endpoint (mock pipeline & cache).
   - Integration tests med TestClient + sample segments.
   - Error path tests (LLM fail, invalid input, cache miss, large payload).
   - Property-based tests med hypothesis p├Ñ schemas.
   - Contract tests: verifiera att response alltid inneh├Ñller f├Ârv├ñntade Fas 4-f├ñlt.
3. K├Âr full suite iterativt efter varje fix: `pytest ... --cov --cov-fail-under=90`.
4. L├ñgg till pre-commit hooks eller CI-steg f├Âr API (om inte finns).

**Deliverables:** ÔëÑ 90 % coverage, alla tester gr├Âna, nya testfiler eller ut├Âkningar.

### Fas 4: Dokumentation & OpenAPI (0,5 dag) ÔÇö **DONE 2026-06-03**
**Levererat:** `docs/API.md` (quickstart, auth, env, endpoints, Fas 4, curl/Python), README-l├ñnk, OpenAPI via `/docs`.  
**Kvar (l├Ñg prio):** `json_schema_extra` examples p├Ñ alla Pydantic-modeller.

**Tasks:**
1. ÔÅ│ Berika alla endpoints med detaljerade docstrings, tags, summaries, descriptions.
2. ÔÅ│ L├ñgg till rika examples i Pydantic-modeller.
3. Ô£à Skapa/uppdatera `docs/API.md` med:
   - Quickstart
   - Exempel f├Âr varje endpoint (curl + Python)
   - Autentisering & rate limiting
   - Error codes
   - Fas 4-specifika anv├ñndningsfall
4. Uppdatera README.md med l├ñnk till docs/API.md.

**Deliverables:** Full OpenAPI + dedikerad dokumentation.

### Fas 5: Slutvalidering, CI & Leverans (0,5 dag) ÔÇö **DONE 2026-06-03**

| Task | Status |
|------|--------|
| Smoke uvicorn (`/health`, `/docs`, `/openapi.json`) | Ô£à |
| API tests 53/53, `src/api` cov **96.38%** | Ô£à |
| ruff `src/api/` | Ô£à |
| CI `api-test` job (ÔëÑ90% gate) | Ô£à |
| Acceptance criteria (plan ┬º2) | Ô£à se tabell nedan |
| Load-test stub | ÔÅ¡´©Å utel├ñmnat (l├Ñg ROI) |
| Merge till `main` | Ô£à lokalt (se git log) |

**Acceptance criteria (┬º2):**

| Kriterium | Uppfyllt |
|-----------|----------|
| Testt├ñckning ÔëÑ90% `src/api` | Ô£à 96% |
| Zero critical P0 (deep_analysis, cache_hit, auth optional) | Ô£à |
| Fas 4 endpoints + caching | Ô£à |
| Dokumentation `docs/API.md` | Ô£à |
| Backward compat (auth off utan env) | Ô£à |

**Release notes (API v0.4.0):**
- Ny s├ñkerhetsmodell: `SENTIMENT_API_KEY`, `API_MEDIA_ROOT`, `X-OpenRouter-Key`
- DI: delad `AggregateCache`, `dependencies.py`, `router_errors.py`
- Fas 4: `deep_analysis` p├Ñ alla requests, payload-gr├ñnser, korrekt `cached`
- 53 API-tester, CI-gate `--cov-fail-under=90`

**Deliverables:** Sign-off, uppdaterad plan, merge-ready kod p├Ñ `api-review-v2-hardening`.

---

## 4. Riskhantering

| Risk                        | Sannolikhet | P├Ñverkan | Mitigation                                      |
|-----------------------------|-------------|----------|-------------------------------------------------|
| Breaking changes            | L├Ñg         | H├Âg      | Anv├ñnd deprecation + version i URL/header       |
| LLM-kostnader via API       | Medel       | Medel    | Rate limit + caching + explicit --use_mistral_llm default False |
| Prestanda p├Ñ stora batcher  | Medel       | H├Âg      | Max limits + caching + async/background         |
| Komplexitet i DI-refaktor   | L├Ñg         | Medel    | B├Ârja med pipeline-dependency, iterera          |
| Otillr├ñcklig test coverage  | Medel       | H├Âg      | Tvingande cov-fail-under + subagent TestAgent   |

---

## 5. Hur man Anv├ñnder Denna Plan med Composer 2.5 / Grok Build

1. Klistra in den "Ultimata prompten" (se separat avsnitt nedan).
2. Composer/spawnade subagents m├Ñste l├ñsa denna fil (`docs/API_REVIEW_HARDENING_PLAN.md`) som prim├ñr k├ñlla.
3. F├Âlj faserna strikt i ordning (0 ÔåÆ 5).
4. Varje fas avslutas med commit + uppdatering av status i denna fil.
5. Anv├ñnd spawn subagent f├Âr parallellt arbete (t.ex. en ImplementationAgent per router + en TestAgent).
6. Iterativ loop: Fix ÔåÆ Test ÔåÆ Validate ÔåÆ Document ÔåÆ Commit.
7. Vid os├ñkerhet: anta production best practice och dokumentera valet i findings.

---

## 6. Bilaga: Redan Fixade & Prioriterade Issues (exempel)

**Redan fixade (live 2026-06-03):**
- `/analyze_pipeline`: `deep_analysis=req.deep_analysis` korrekt.
- **P0-1 (Fas 1):** `/agent_performance` ÔÇö `deep_analysis` var felaktigt `req.use_mistral_llm`; korrigerat + schema + regressionstest.

**H├Âgprioriterade att fixa i Fas 1ÔÇô2:**
- Saknade `deep_analysis` f├ñlt i Fas 4 request schemas.
- Breda `except Exception` ÔåÆ anv├ñnd specifika domain errors.
- Inkonsekvent llm_api_key-hantering.
- Avsaknad av DI, rate limiting, CORS, tracing.
- L├Ñg testt├ñckning f├Âr nya endpoints.
- Ineffektiv pipeline-instansiering i varje request.

---

**Slut p├Ñ plan v2.0**  
Denna plan ├ñr nu den officiella, uppdaterade och f├Ârb├ñttrade versionen. Alla framtida arbete p├Ñ API:n ska referera till och f├Âlja denna plan.
