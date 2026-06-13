# Refaktoreringsplan: Backend & API – Robusthet, Stabilitet och Kodkvalitet (Post-Fas 4)

**Projekt:** Automatisk sentimentanalys – Svenskt Call Center Intelligence-system  
**Version:** 1.0  
**Datum:** 2026-06-13  
**Status:** Planerad – redo för implementation  
**Målgrupp:** Utvecklare, Grok Build / Windsurf-agenter, QA  

---

## 1. Bakgrund och nuläge

Som en del av det pågående projektet att utveckla ett svenskt sentimentanalys-system för call center (med ASR, speaker diarization, intent-klassificering, aspect-based sentiment, emotion detection, trajectory-analys, Mistral/OpenRouter LLM-integration och Fas 4 call center features som agent performance, compliance/QA scoring, insights aggregator, semantic search, PII redaction och alerting) har vi implementerat en solid grund i backend och REST API (FastAPI).

Fas 4 har levererat många nya moduler (`src/agent_performance.py`, `src/compliance_qa.py`, `src/insights_aggregator.py`, `src/semantic_search.py`, `src/alerting.py`, `src/caching.py`, utökningar i `src/pipeline.py`, `src/llm/*`, `src/api/routers/pipeline.py` m.fl.) och API-endpoints. 

**Dock visar omfattande testing (se TESTING_STATUS_BACKEND_API.md) att systemet fortfarande har instabilitet:**
- Edge cases och error paths hanteras inte alltid graceful.
- Potentiell duplicerad/onödig kod i routers, helpers och pipeline-logik.
- Tight coupling mellan HTTP-layer och core analysis.
- Ofullständig validering och observability.
- Testflakiness och luckor i integrationstester för de nya Fas 4-funktionerna.

Målet med denna refaktorering är att lyfta backend/API från "fungerar mestadels" till **robust, clean, stabil och production-ready** utan att förlora funktionalitet eller backward compatibility.

## 2. Övergripande mål
- **Robusthet:** Bättre felhantering, retry/circuit breaker, graceful degradation (local fallback vid LLM-fel), stark input validation.
- **Ta bort onödig kod:** Dead code, duplication, överkomplicerad logik, legacy-från Streamlit-migrering.
- **Stabilitet:** Färre runtime errors, consistent beteende, hög test coverage, load/resilience testing.
- **Bättre kodkvalitet:** Läsbarhet, modularitet, maintainability, följande FastAPI/Pydantic best practices + svensk call center-domän.
- **GDPR & Privacy by design:** Fortsätt stärka PII-hantering och audit.
- **Förberedelse för skalning:** Bättre async, caching, metrics för framtida dashboard/real-time.

## 3. Nulägesanalys (efter grundlig genomgång)

**Styrkor:**
- Ren FastAPI-struktur med lifespan, middleware (RequestId + security headers), custom exception handlers.
- Pydantic schemas för request/response.
- Hybrid local + LLM arkitektur med caching.
- Nya Fas4 moduler väl integrerade i pipeline och API.

**Svagheter & risker för instabilitet:**
- Stora monolitiska filer (pipeline.py ~32kB, schemas.py ~19kB, vissa routers).
- Duplicerad logik mellan endpoints (analyze vs batch vs scan).
- Error handling bra på hög nivå men saknar djup i pipeline-steps och partial failures.
- Begränsad structured logging och metrics.
- Vissa paths saknar storleksbegränsningar eller rate limiting.
- Test coverage ojämn – nya features har bra unit men integration/concurrency luckor.
- Potentiella race conditions eller state issues i shared cache/alert_engine.
- Onödiga imports eller över-engineered delar i nya moduler.

## 4. Refaktoreringsplan – Fasindelad (totalt ~3–4 veckor, inkrementell)

### Fas 0: Audit & Baseline (1–2 dagar)
- Statisk analys: `ruff check --fix`, `ruff format`, `mypy src/api src/pipeline.py`, `vulture` för dead code, complexity metrics.
- Full inventering av endpoints, Pydantic models, dependencies.
- Baseline: kör alla tester (`pytest -m "not slow"`), coverage report, dokumentera failing/flaky tests.
- Identifiera specifik onödig kod och duplication (t.ex. i helpers.py, routers).
- **Deliverable:** Audit-rapport + TODO-lista i GitHub Issues.

### Fas 1: Kodstädning & Ta bort onödig kod (2–3 dagar)
- Ta bort dead code, unused imports, commented-out legacy kod.
- Konsolidera duplicerad kod till shared helpers/services (t.ex. gemensam `_run_analysis_flow`).
- Bryt ner stora funktioner (t.ex. i pipeline.py och routers/pipeline.py).
- Förenkla överkomplicerade delar i Fas4-moduler där möjligt (keep it simple).
- Uppdatera pyproject.toml / ruff config om nödvändigt.
- **Deliverable:** Clean commit(s), ruff clean build.

### Fas 2: Robusthet & Avancerad Felhantering (4–5 dagar)
- Utöka `src/core/errors.py` med fler specifika exceptions (t.ex. `PartialBatchError`, `LLMTimeoutError`, `ValidationError`).
- Standardisera alla responses till strukturerad form: `{ "success": bool, "data": ..., "error": { "code": str, "message": str, "details": ..., "request_id": str } }`.
- Implementera retry med exponential backoff + circuit breaker för LLM calls och transcription backends (använd `tenacity` eller liknande).
- Hantera partial failures i batch/scan endpoints (per-item status + summary).
- Lägg till request size limits, timeout middleware, rate limiting (t.ex. `slowapi` eller custom).
- Graceful degradation: om LLM eller extern tjänst fallerar → fallback till lokal analys + warning i report + loggad incident.
- Stärk PII-redaction validering och error paths.
- **Deliverable:** Alla kritiska endpoints robusta mot edge cases, uppdaterade tester.

### Fas 3: Modularitet & Service Layer (4–5 dagar)
- Inför `src/api/services/` (t.ex. `pipeline_service.py`, `analysis_service.py`).
- Flytta business logic från routers till services – routers blir tunna (HTTP + auth + response mapping).
- Förbättra dependency injection i `dependencies.py` (fler providers för cache, alert_engine, profile, redactor).
- Decouple: routers → services → pipeline/core.
- Event-driven förbättringar (utnyttja befintlig TranscriptionEventHub).
- Överväg enkel registry för analyzers för lättare extension.
- **Deliverable:** Tydligare arkitektur, enklare att testa och utöka.

### Fas 4: Stabilitet, Prestanda & Observability (3–4 dagar)
- Structured logging (structlog eller JSON logs med request_id, duration, component, user_context).
- Lägg till Prometheus metrics middleware + custom metrics (analyze_duration, llm_calls, cache_hit_rate, error_rate per endpoint).
- Optimera caching: smartare invalidation, background precompute jobs, Redis best practices.
- Async-förbättringar: flytta CPU-tunga sync calls till `run_in_threadpool` där lämpligt.
- Setup för load testing (t.ex. Locust scripts i `tests/load/`).
- Utökade health checks (cache ready, models loaded, external LLM connectivity).
- **Deliverable:** Mätbara prestanda + observability, stabilt under load.

### Fas 5: Testing, CI & Kvalitet (2–3 dagar + ongoing)
- Utöka `tests/test_api.py`, contract tests, snapshot testing för responses.
- Lägg till property-based testing (Hypothesis) för schemas och pipelines.
- Full end-to-end integration tests med samples + mocked LLM/external.
- Concurrency & chaos testing för error resilience.
- Uppdatera CI (GitHub Actions) med nya linters, coverage gates.
- Uppdatera `TESTING_STATUS_BACKEND_API.md` med ny status.
- **Deliverable:** >85% coverage på api+core, alla tester stabila.

### Fas 6: Dokumentation, Versioning & Release (1–2 dagar)
- Uppdatera `docs/API.md`, OpenAPI-exempel, README.
- Inför API versioning (t.ex. prefix `/v1/`) om breaking changes.
- Uppdatera CHANGELOG.md och UTVECKLINGSPLAN.md.
- Skapa release notes + migreringsguide.
- **Deliverable:** Produktionsklar dokumentation.

## 5. Specifika fokusområden (hög prioritet)
- `src/api/app.py`: Utöka middleware, förbättra lifespan init, lägg till metrics.
- `src/api/routers/*.py` (särskilt pipeline.py, transcription.py): Förenkla, extrahera till services.
- `src/api/schemas.py`: Eventuell split till flera filer (pipeline, qa, insights) om den blir för stor.
- `src/pipeline.py`: Modularisering av `CallAnalysisPipeline` (t.ex. Orchestrator + steps).
- `src/api/dependencies.py` & `helpers.py`: Stärk och centralisera.
- Nya moduler: `src/api/services/`, `src/api/exceptions.py` (eller utöka core/errors), `src/core/logging_config.py`.
- Bevara & förstärk: PII redaction early, GDPR-loggning, hybrid fallback.

## 6. Framgångskriterier (mätbara)
- Ruff + mypy + format clean på alla ändrade filer.
- 100% befintliga tester passerar + nya tester för edge cases.
- Inga ohanterade exceptions i normala + fel-scenarier (simulerade).
- API responses alltid konsistenta och strukturerade.
- Prestanda: p95 latency för `/analyze_pipeline` förbättrad eller oförändrad.
- Kodtäckning > 85% för `src/api/` och `src/pipeline.py`.
- Enkel att lägga till ny feature/analyzer utan att ändra routers.
- Dokumenterad arkitektur i `docs/ARCHITECTURE.md` (uppdatera).

## 7. Risker & hantering
- **Regression:** Omfattande regressionstester + feature flags för nya felhanteringsvägar.
- **Breaking changes:** Börja med backward-compatible ändringar; deprecate gamlas vägar.
- **Tid & omfattning:** Prioritera Fas 0-2 först (hög ROI på stabilitet).
- **Komplexitet:** Håll refaktorn inkrementell – en fas i taget, commit ofta.

## 8. Hur man använder denna plan
1. Läs in hela planen + relaterade filer (README, TESTING_STATUS_BACKEND_API.md, UTVECKLINGSPLAN.md, src/api/app.py, src/pipeline.py).
2. Börja med Fas 0 audit.
3. Använd dedikerad Grok Build-prompt (se separat) för implementation.
4. Efter varje fas: kör tester, uppdatera status, commit + push.
5. Iterera med sub-agents för testing/verifiering.

**Denna plan är en naturlig fortsättning på Fas 4 och gör systemet redo för produktion och vidareutveckling av call center intelligence-funktioner.**

---

*Skapad som del av projektet Automatisk sentimentanalys för svenskt call center.*