# AGENT CONTEXT — Automatisk-sentimentanalys
**Generated:** 2026-06-26 | Use this file as the single source of truth when continuing development. Re-run github-project-status skill after significant changes.

## 1. What This System Is
Svenskt Call Center Intelligence-system för automatisk sentimentanalys, transkribering (ASR + diarization), intent/ emotion/ aspect-analys, LLM-baserad QA/compliance, PII-redaktion, insikter, agent performance metrics och realtids-dashboard. Byggt för svenska kundtjänstsamtal med stark GDPR-fokus, skalbarhet och self-hosted/VPS-deployment. Mål: ersätta manuell samtalshantering med automatiserad, pålitlig analys som ger actionable insights till QA, coacher och chefer.

**Core Value**: End-to-end från ljud → strukturerad rapport (sentiment scores, intents, QA-evidence, alerts, hot topics, coaching tips) → dashboard/API consumption. Stödjer batch, real-time WS och interaktiv NiceGUI UI.

## 2. Current Feature Inventory

**Implemented / Live**:
- ASR/Transcription: faster-whisper, WhisperX, diarization, Swedish models, preprocess, batch + WebSocket real-time. Launcher för Windows ASR-hantering.
- Analysis Pipeline: Registry-baserad CallAnalysisPipeline med sentiment, emotion, intent, aspect, topics, trajectory, llm_judge (partial), pii_redactor (hardened), negation, blending. 
- LLM Layer: Pluggable providers (Groq ny med EU-residency gate + caching + cost, Mistral, OpenRouter). Strict Pydantic schemas, prompts, fallback chains.
- Fas 4 Backend: agent_performance, compliance_qa (YAML scorecards + LLM hybrid + evidence spans), insights_aggregator (hot topics, root cause, trender), semantic_search (hybrid vector+keyword), alerting (rules + webhook w/ circuit breaker + retries).
- API (FastAPI): Routers för transcription, pipeline, scan, text, ws, batch, agent_performance, search/semantic, qa/score, alerts. Schemas, middleware, services, cached responses, OpenAPI.
- Dashboard (NiceGUI): KPI, calls table, live_analysis, transcription_monitor, call_detail (emotion timeline, virtual transcript), hot_topic_wordcloud, insights_hot_topics, agent_performance, qa_scorecard, alerts_panel, pii_audit, test_lab, fas4_insights. Local demo + API client modes. Modern, responsive UI.
- CLI + Evaluation: Full CLI, evaluate (fas4-validation reports), audio benchmarks, finetune scripts, data prep.
- Data: callcenter_train/val (CSV/JSONL), sensaldo_lexicon, intent data; reports/.
- Security: PII hardening (Luhn, Swedish names/phones/addresses), GDPR LLM routing, SECURITY.md, preflight.
- Infra: pyproject.toml optional deps, Docker, Makefile, configs (llm, alerting, qa_scorecards), Windows installer/launcher, tests (509+ , high src/ coverage).

**In Progress**:
- LLM-judge full wiring into pipeline + dashboard visualizations.
- Alerting production polish (mark handled, test webhook in UI).
- Dashboard refinements (async, refresh, edge cases from Fas4).

**Planned**:
- Expanded finetuning/production models.
- More LLM options or local inference.
- Production deployment automation, scaling.
- Deferred: YouTube ingest, some stubs.

## 3. Architecture & Key Components
- **Pipeline Core** (`src/pipeline.py` + `src/analysis/registry.py`): Orchestrates steps. Steps registered, run in sequence/parallel where safe. Results merged into report. Supports caching via `src/caching.py` (AggregateCache).
- **Transcription** (`src/transcription/factory.py`, `faster_whisper.py`, `whisperx.py` etc.): Backend abstraction. Diarization separate. Preprocess for Swedish audio.
- **Analysis** (`src/analysis/`, `src/sentiment.py`, `src/intent.py`, `src/llm_judge.py` etc.): Modular, each returns structured dict/Pydantic. LLM judge for advanced QA/evidence.
- **LLM** (`src/llm/`): Clients + Analyzers per provider. `groq_client.py`, `mistral_analyzer.py`, `openrouter_client.py`. Schemas define output strictly. Prompts in `prompts.py`. PII redaction before LLM calls where configured.
- **API** (`src/api/app.py`, `routers/`, `services/`): Dependency injection, rate limit, error responses, transcription_jobs, pipeline_cache. WS for streaming transcription events.
- **Dashboard** (`app/nicegui_dashboard/`): Components in `components/`, services in `services/` (nicegui_api_client.py for backend calls, fas4_data.py for local, chart_data.py). State management, theme, layout. Test pages in tests/fixtures.
- **Launcher** (`launcher/`): Process manager, ASR dialog, status panel, env builder, pid store. PowerShell entry for desktop users.
- **Data Layer**: Local CSV/JSONL + in-memory/demo providers. Caching for expensive aggregations.
- **Key Invariants**: Always PII-safe (redact before LLM if flag set), graceful degradation (missing optional deps don't crash core), Swedish-first (lexicon, prompts, data), structured output everywhere (Pydantic), tests cover happy + edge paths.

**Design Decisions**: Registry pattern for extensibility (easy add new analysis step). Provider abstraction for LLM. Hybrid rule+LLM for QA. Pre-compute + cache for dashboard speed. Windows launcher to lower barrier for non-dev users.

## 4. Important File Map
- `README.md`, `CHANGELOG.md`, `ROADMAP.md`, `UTVECKLINGSPLAN.md`, `docs/ARCHITECTURE.md`, `docs/LLM_AGENT_GUIDE.md`, `docs/FAS4_COMPLETION.md` — Read these first on any new session.
- `src/pipeline.py` — Core orchestration; know the step order and result merging.
- `src/analysis/registry.py` — How to register new analyzers; patterns for graceful handling.
- `src/llm/schemas.py` + `src/llm/prompts.py` — LLM output contracts and prompt engineering.
- `src/api/routers/pipeline.py` + `src/api/schemas.py` — API contract for analysis requests.
- `app/nicegui_dashboard/main.py` + `app/nicegui_dashboard/services/nicegui_api_client.py` — Dashboard entry + data fetching.
- `launcher/main.py` + `launcher/process_manager.py` — Desktop launcher logic.
- `pyproject.toml` — Dependencies, optional groups (cli, api, dashboard-nicegui, diarize), scripts.
- `configs/` — llm_config.yaml, alerting_config.yaml, qa_scorecards/*.yaml, install_defaults.
- `tests/` — Extensive; run with pytest. Many test_*.py mirroring src/.
- `docs/LLM_AGENT_GUIDE.md` — **Most important for agents**: architecture philosophy, patterns, what to do/not do, security rules.

## 5. How to Work With This Codebase
- **Setup**: `pip install -e "[cli,api,dashboard-nicegui]"` then `sentimentanalys download-asr`. For dev: requirements-dev.txt + pre-commit.
- **Run**:
  - CLI: `sentimentanalys --help` or `python -m src.cli`
  - API: `uvicorn src.api:app --reload` (or via launcher)
  - Dashboard: `python -m app.nicegui_dashboard.main`
  - Windows: `\.\launcher.ps1` or `Sentimentanalys.bat`
  - Tests: `pytest` (or specific `pytest tests/test_pipeline.py -q`)
  - Evaluate: `python -m src.evaluate fas4-validation`
- **Coding Conventions**: Strict Pydantic models for all I/O and LLM output. Type hints everywhere. Logging via standard. Error handling with custom exceptions in `src/core/errors.py`. Swedish variable/docstrings where domain-specific. Pre-commit hooks (ruff, mypy?).
- **Adding Features**:
  - New analysis step: Implement in `src/analysis/`, register in registry.py, add to pipeline, update schemas/tests.
  - New LLM provider: Add client + analyzer in `src/llm/`, update LLM_PROVIDERS.md and schemas.
  - Dashboard tab/component: New file in `components/`, wire in main/layout, add to api_client if backend.
  - API endpoint: Router in `routers/`, service if needed, schema, test.
- **Testing**: Unit + integration. Mock external (LLM, ASR heavy). Use fixtures. Aim high coverage on src/.
- **Docs**: Update ROADMAP/CHANGELOG on releases. Use this AGENT_CONTEXT + PROJECT_STATUS as single source. Run github-project-status skill after sessions.

**Do NOT**: Break PII redaction or GDPR paths. Ignore Swedish language specifics. Assume GPU always present (graceful CPU fallback). Commit without tests/docs update.

## 6. Open Tasks & Priorities
- **High Priority**: Complete LLM-judge integration (remove stubs), wire to dashboard + API. Polish Fas4 insights/hot topics/alerts UI. Full alerting webhook + test in test_lab.
- **Next**: Expand evaluation reports, finetune on more data, prepare v0.5 release. Dashboard performance (cache hit visibility, async).
- **Backlog**: Production deployment (Docker/K8s?, secrets mgmt), additional providers, mobile support, long-form/YouTube if revived.
- **Invariants / Do not break**: PII redaction order and validation (Luhn before phone etc.), EU-residency gate for Groq, structured Pydantic output from LLM, cache metadata semantics, test contracts in tests/contracts/.

## 7. Context for Future Agents
- This is a mature, production-oriented system with strong test coverage and documentation. Fas 1-4 backend mostly complete; focus now on LLM-judge polish, dashboard UX and release.
- Always start new session by reading: `docs/LLM_AGENT_GUIDE.md`, `PROJECT_STATUS.md` (this file's sibling), `ROADMAP.md`, `SECURITY.md`.
- Recent changes (Groq, PII harden, Fas4 dashboard) show pattern: add provider/analyzer → update schemas/prompts → API/dashboard wiring → tests + docs.
- Use Swedish terminology in UI/docs where appropriate (kundtjänst, samtal, transkribering, etc.).
- For image gen or visuals: project uses generated images for plans/home but core is code+data.
- Re-run `github-project-status` skill (or equivalent) after any coding/refactor session to keep docs in sync — this prevents context drift for future agents.
- The project aligns with broader goals of Swedish AI tooling for practical business use (e-comm automation, call center intelligence).

**When handing off or starting fresh**: Paste this AGENT_CONTEXT.md + repo URL into the prompt. New agent can immediately continue without re-explaining architecture.