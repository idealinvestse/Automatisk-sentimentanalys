# AGENT CONTEXT — Automatisk-sentimentanalys
**Generated:** 2026-07-09 | **Version:** 0.5.0 | Use this file as the single source of truth when continuing development.

## 1. What This System Is
Svenskt Call Center Intelligence-system för automatisk sentimentanalys, transkribering (ASR + diarization), intent/ emotion/ aspect-analys, LLM-baserad QA/compliance, PII-redaktion, insikter, agent performance metrics och realtids-dashboard. Byggt för svenska kundtjänstsamtal med stark GDPR-fokus, skalbarhet och self-hosted/VPS-deployment. Mål: ersätta manuell samtalshantering med automatiserad, pålitlig analys som ger actionable insights till QA, coacher och chefer.

**Core Value**: End-to-end från ljud → strukturerad rapport (sentiment scores, intents, QA-evidence, alerts, hot topics, coaching tips) → dashboard/API consumption. Stödjer batch, real-time WS och interaktiv Next.js web UI. **Ny**: Dynamisk LLM model pricing via OpenRouter catalog.

## 2. Current Feature Inventory

**Implemented / Live**:
- ASR/Transcription: faster-whisper, WhisperX, diarization, Swedish models, preprocess, batch + WebSocket real-time. Launcher för Windows ASR-hantering. **Ny**: Valbar mapp för modeller/downloads.
- Analysis Pipeline: Registry-baserad CallAnalysisPipeline med sentiment, emotion, intent, aspect, topics, trajectory, llm_judge, pii_redactor, negation, blending.
- LLM Layer: Pluggable providers (Groq med EU-residency gate + caching + cost, Mistral, OpenRouter). Strict Pydantic schemas, prompts, fallback chains. **Ny**: Dynamic pricing från model_catalog + refresh_pricing_from_catalog().
- Fas 4 Backend: agent_performance, compliance_qa (YAML scorecards + LLM hybrid + evidence spans), insights_aggregator (hot topics, root cause, trender), semantic_search (hybrid vector+keyword), alerting (rules + webhook w/ circuit breaker + retries).
- API (FastAPI): Routers för transcription, pipeline, scan, text, ws, batch, agent_performance, search/semantic, qa/score, alerts. Schemas, middleware, services, cached responses, OpenAPI.
- Dashboard (webui/ — Next.js 16 + React 19 + TS + Tailwind v4): KPI, calls table, call_detail (emotion timeline, QA scorecard, alerts, LLM judge), analytics, agent_performance, fas4_insights (hot topics + alerts panel), transcription (WS), test_lab. All data from real backend pipeline via React Query. **Legacy**: `app/archive/nicegui_dashboard/` är deprecated, se `docs/WEBUI_MODERNIZATION_PLAN.md`.
- CLI + Evaluation: Full CLI, evaluate (fas4-validation reports), audio benchmarks, finetune scripts, data prep. **Ny**: `scan-openrouter-models` kommando med rich table.
- Data: callcenter_train/val (CSV/JSONL), sensaldo_lexicon, intent data; reports/.
- Security: PII hardening (Luhn, Swedish names/phones/addresses), GDPR LLM routing, SECURITY.md, preflight.
- Infra: pyproject.toml optional deps, Docker, Makefile, configs (llm_config.yaml, alerting_config.yaml, qa_scorecards/*.yaml, install_defaults).
- Fine-tuning: src/fine_tuning/ + integration in dashboard model selector + live training support.
- Real-time & Production: WebSocket, persistent alerting state (JSON), multi-worker ready, Docker/VPS support.
- **LLM Model Management (Ny)**: `src/llm/model_catalog.py` (full OpenRouter scan + save), dynamic pricing i openrouter_client, CLI + Dashboard integration.
- **Model A/B compare (v0.5)**: `POST /analyze_pipeline/compare` (max 3 models, budget guard) + `webui/src/components/model-compare-panel.tsx` in Testlabb.
- **DATA-01 import (v0.5)**: `scripts/import_domain_corpus.py`, gitignored `data/import/` slot, runbook in `docs/DEVELOPMENT.md`.
- **Docker staging (v0.5)**: `docker-compose.staging.yml` (api+webui+redis+prometheus), `scripts/staging_observability_smoke.py`.
- **Grok Build Optimization**: `.grok/skills/` med 6 custom skills (github-project-status, grok-repo-optimizer, github-repo-deep-dive, code-review-reflector, grok-full-launcher, repo-health-check) + enhanced AGENTS.md + Grok Build quickstart i README.

**In Progress**:
- Real annotated call corpus import (workflow ready, awaiting external data).
- Intent fine-tune model beating heuristic + 0.05 macro F1.

**Planned**:
- Expanded finetuning/production models.
- Voice synthesis and multi-modal features.
- Edge AI Network and customer expansion (DK).
- Model picker + auto cost-optimized routing baserat på catalog.

## 3. Architecture & Key Components
- **Pipeline Core** (src/pipeline.py + src/analysis/registry.py): Orchestrates steps. Steps registered, run in sequence/parallel where safe. Results merged into report. Supports caching via src/caching.py (AggregateCache).
- **Transcription** (src/transcription/factory.py, faster_whisper.py, whisperx.py etc.): Backend abstraction. Diarization separate. Preprocess for Swedish audio.
- **Analysis** (src/analysis/, src/sentiment.py, src/intent.py, src/analysis/llm_judge.py etc.): Modular, each returns structured dict/Pydantic. LLM judge for advanced QA/evidence.
- **LLM** (src/llm/): Clients + Analyzers per provider. groq_client.py, mistral_analyzer.py, openrouter_client.py (nu med dynamic pricing från catalog). Schemas define output strictly. Prompts in prompts.py. PII redaction before LLM calls where configured.
- **API** (src/api/app.py, routers/, services/): Dependency injection, rate limit, error responses, transcription jobs, upload, edge router, pipeline_cache. WS for streaming transcription events (+ ticket endpoint when auth enabled).
- **Dashboard** (webui/): Next.js App Router pages, React Query hooks, typed API client (`webui/src/lib/api/client.ts`). shadcn/ui primitives + feature components. **Legacy**: `app/archive/nicegui_dashboard/` (Python/NiceGUI) är deprecated men behålls som referens.
- **Launcher** (launcher/): Process manager, ASR dialog, status panel, env builder, pid store. PowerShell entry for desktop users. **Ny**: Valbar mapp för modeller.
- **Data Layer**: Local CSV/JSONL + in-memory/demo providers. Caching for expensive aggregations.
- **Key Invariants**: Always PII-safe (redact before LLM if flag set), graceful degradation (missing optional deps don't crash core), Swedish-first (lexicon, prompts, data), structured output everywhere (Pydantic), tests cover happy + edge paths. **Ny**: Model catalog pricing är live och uppdateras via CLI/Dashboard.
- **Grok Build Specific**: .grok/skills/ + AGENTS.md gör repot omedelbart användbart i Grok Build utan extra setup.

**Design Decisions**: Registry pattern for extensibility (easy add new analysis step). Provider abstraction for LLM. Hybrid rule+LLM for QA. Pre-compute + cache for dashboard speed. Windows launcher to lower barrier for non-dev users. **Ny**: Central model catalog för kostnadskontroll och smart model-val.

## 4. Important File Map
- README.md, CHANGELOG.md, docs/ROADMAP.md, docs/CLEANUP_PLAN.md, docs/ARCHITECTURE.md, docs/LLM_AGENT_GUIDE.md, docs/FAS4_COMPLETION.md — Read these first on any new session.
- src/pipeline.py — Core orchestration; know the step order and result merging.
- src/analysis/registry.py — How to register new analyzers; patterns for graceful handling.
- src/llm/schemas.py + src/llm/prompts.py — LLM output contracts and prompt engineering.
- src/llm/openrouter_client.py + src/llm/model_catalog.py — **Ny viktig**: Dynamic pricing + full model scan. Använd refresh_pricing_from_catalog() och load_catalog().
- src/api/routers/pipeline.py + src/api/schemas.py — API contract for analysis requests; **v0.5:** `POST /analyze_pipeline/compare`.
- webui/src/lib/api/client.ts + webui/src/hooks/ — Primary frontend API client + React Query hooks. **Legacy**: app/archive/nicegui_dashboard/ (deprecated, referens endast).
- launcher/main.py + launcher/process_manager.py — Desktop launcher logic. **Ny**: Storage path settings.
- pyproject.toml — Dependencies, optional groups (cli, api, dashboard-nicegui, diarize), scripts.
- configs/ — llm_config.yaml, alerting_config.yaml, qa_scorecards/*.yaml, install_defaults.
- tests/ — **921** test functions (`pytest --collect-only`); run with pytest. Many test_*.py mirroring src/.
- **.grok/skills/** — Custom skills (github-project-status, grok-repo-optimizer, github-repo-deep-dive, code-review-reflector, grok-full-launcher, repo-health-check) — använd dessa direkt i Grok Build.
- docs/LLM_AGENT_GUIDE.md — **Most important for agents**: architecture philosophy, patterns, what to do/not do, security rules.

## 5. How to Work With This Codebase
- **Setup**: pip install -e ".[cli,api,dashboard-nicegui,install]" then sentimentanalys download-asr . For dev: pip install -e ".[dev,diarize]" + pre-commit.
- **Run**:
  - CLI: sentimentanalys --help or python -m src.cli   (ny: scan-openrouter-models)
  - API: uvicorn src.api:app --reload (or via launcher)
  - Dashboard (primär): cd webui && npm run dev   (Next.js, http://localhost:3000)
  - Dashboard (legacy, ej underhåll): python -m app.archive.nicegui_dashboard.main
  - Windows: .\launcher.ps1 or Sentimentanalys.bat   (ny: valbar modeller-mapp)
  - Tests: pytest (or specific pytest tests/test_pipeline.py -q )
  - Evaluate: python -m src.evaluate fas4-validation
- **Coding Conventions**: Strict Pydantic models for all I/O and LLM output. Type hints everywhere. Logging via standard. Error handling with custom exceptions in src/core/errors.py. Swedish variable/docstrings where domain-specific. Pre-commit hooks (ruff, mypy?).
- **Adding Features**:
  - New analysis step: Implement in src/analysis/, register in registry.py, add to pipeline, update schemas/tests.
  - New LLM provider / model handling: Använd model_catalog.py + uppdatera openrouter_client pricing.
  - Dashboard tab/component: New page in webui/src/app/, hook in webui/src/hooks/, component in webui/src/components/. Använd React Query + typed API client.
  - API endpoint: Router in routers/, service if needed, schema, test.
- **Testing**: Unit + integration. Mock external (LLM, ASR heavy). Use fixtures. Aim high coverage on src/. Canonical runbook: docs/DEVELOPMENT.md § Testing (L0–L9); release L7–L9 in docs/PRODUCTION_CHECKLIST.md.
- **Docs**: Update ROADMAP/CHANGELOG on releases. Use this AGENT_CONTEXT + PROJECT_STATUS as single source. **Re-run github-project-status skill after significant changes**.
- **Grok Build**: Använd skills i .grok/skills/ direkt (t.ex. "github-project-status skill" eller "code-review-reflector").

## 6. Open Tasks & Priorities
- **Post v0.5:** Real corpus via DATA-01 import; intent fine-tune; OTLP tracing in production.
- Model picker + auto cost-optimized routing baserat på catalog.
- Fas 6 commercialization (post v0.5).

## 7. Context for Future Agents
After every change that affects features, re-run the github-project-status skill. We use Swedish/Norwegian localization in UI and prompts. Strong focus on PII/GDPR and graceful degradation. When implementing new analyzer or LLM feature (särskilt model catalog / pricing), se src/llm/model_catalog.py och openrouter_client.py. Dashboard components should use `webui/src/lib/api/client.ts` for backend data (NiceGUI-dashboarden är deprecated). Använd .grok/skills/ för kvalitet, review och launch-hjälp. **Ny regel**: Efter model-relaterade ändringar, kör `sentimentanalys scan-openrouter-models` och uppdatera pricing i client.
