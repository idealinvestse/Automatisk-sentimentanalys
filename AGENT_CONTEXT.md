# AGENT CONTEXT — Automatisk-sentimentanalys
**Generated:** 2026-06-27 | Use this file as the single source of truth when continuing development.

## 1. What This System Is
Svenskt Call Center Intelligence-system för automatisk sentimentanalys, transkribering (ASR + diarization), intent/ emotion/ aspect-analys, LLM-baserad QA/compliance, PII-redaktion, insikter, agent performance metrics och realtids-dashboard. Byggt för svenska kundtjänstsamtal med stark GDPR-fokus, skalbarhet och self-hosted/VPS-deployment. Mål: ersätta manuell samtalshantering med automatiserad, pålitlig analys som ger actionable insights till QA, coacher och chefer.

**Core Value**: End-to-end från ljud 	o strukturerad rapport (sentiment scores, intents, QA-evidence, alerts, hot topics, coaching tips) 	o dashboard/API consumption. Stödjer batch, real-time WS och interaktiv NiceGUI UI.

## 2. Current Feature Inventory

**Implemented / Live**:
- ASR/Transcription: faster-whisper, WhisperX, diarization, Swedish models, preprocess, batch + WebSocket real-time. Launcher för Windows ASR-hantering.
- Analysis Pipeline: Registry-baserad CallAnalysisPipeline med sentiment, emotion, intent, aspect, topics, trajectory, llm_judge, pii_redactor, negation, blending.
- LLM Layer: Pluggable providers (Groq med EU-residency gate + caching + cost, Mistral, OpenRouter). Strict Pydantic schemas, prompts, fallback chains.
- Fas 4 Backend: agent_performance, compliance_qa (YAML scorecards + LLM hybrid + evidence spans), insights_aggregator (hot topics, root cause, trender), semantic_search (hybrid vector+keyword), alerting (rules + webhook w/ circuit breaker + retries).
- API (FastAPI): Routers för transcription, pipeline, scan, text, ws, batch, agent_performance, search/semantic, qa/score, alerts. Schemas, middleware, services, cached responses, OpenAPI.
- Dashboard (NiceGUI): KPI, calls table, live_analysis, transcription_monitor, call_detail (emotion timeline, virtual transcript), hot_topic_wordcloud, insights_hot_topics, agent_performance, qa_scorecard, alerts_panel, pii_audit, test_lab, fas4_insights. Local demo + API client modes. Modern, responsive UI.
- CLI + Evaluation: Full CLI, evaluate (fas4-validation reports), audio benchmarks, finetune scripts, data prep.
- Data: callcenter_train/val (CSV/JSONL), sensaldo_lexicon, intent data; reports/.
- Security: PII hardening (Luhn, Swedish names/phones/addresses), GDPR LLM routing, SECURITY.md, preflight.
- Infra: pyproject.toml optional deps, Docker, Makefile, configs (llm, alerting, qa_scorecards), Windows installer/launcher, tests (500+ , high src/ coverage).
- Fine-tuning: src/fine_tuning/ + integration in dashboard model selector + live training support.
- Real-time & Production: WebSocket, persistent alerting state (JSON), multi-worker ready, Docker/VPS support.

**In Progress**:
- Full production fine-tuning training loop and evaluation.
- v0.5 release preparation.

**Planned**:
- Expanded finetuning/production models.
- Voice synthesis and multi-modal features.
- Edge AI Network and customer expansion (DK).

## 3. Architecture & Key Components
- **Pipeline Core** (src/pipeline.py + src/analysis/registry.py): Orchestrates steps. Steps registered, run in sequence/parallel where safe. Results merged into report. Supports caching via src/caching.py (AggregateCache).
- **Transcription** (src/transcription/factory.py, faster_whisper.py, whisperx.py etc.): Backend abstraction. Diarization separate. Preprocess for Swedish audio.
- **Analysis** (src/analysis/, src/sentiment.py, src/intent.py, src/llm_judge.py etc.): Modular, each returns structured dict/Pydantic. LLM judge for advanced QA/evidence.
- **LLM** (src/llm/): Clients + Analyzers per provider. groq_client.py, mistral_analyzer.py, openrouter_client.py. Schemas define output strictly. Prompts in prompts.py. PII redaction before LLM calls where configured.
- **API** (src/api/app.py, routers/, services/): Dependency injection, rate limit, error responses, transcription_jobs, pipeline_cache. WS for streaming transcription events.
- **Dashboard** (app/nicegui_dashboard/): Components in components/, services in services/ (nicegui_api_client.py for backend calls, fas4_data.py for local, chart_data.py). State management, theme, layout. Test pages in tests/fixtures.
- **Launcher** (launcher/): Process manager, ASR dialog, status panel, env builder, pid store. PowerShell entry for desktop users.
- **Data Layer**: Local CSV/JSONL + in-memory/demo providers. Caching for expensive aggregations.
- **Key Invariants**: Always PII-safe (redact before LLM if flag set), graceful degradation (missing optional deps don't crash core), Swedish-first (lexicon, prompts, data), structured output everywhere (Pydantic), tests cover happy + edge paths.

**Design Decisions**: Registry pattern for extensibility (easy add new analysis step). Provider abstraction for LLM. Hybrid rule+LLM for QA. Pre-compute + cache for dashboard speed. Windows launcher to lower barrier for non-dev users.

## 4. Important File Map
- README.md, CHANGELOG.md, ROADMAP.md, UTVECKLINGSPLAN.md, docs/ARCHITECTURE.md, docs/LLM_AGENT_GUIDE.md, docs/FAS4_COMPLETION.md — Read these first on any new session.
- src/pipeline.py — Core orchestration; know the step order and result merging.
- src/analysis/registry.py — How to register new analyzers; patterns for graceful handling.
- src/llm/schemas.py + src/llm/prompts.py — LLM output contracts and prompt engineering.
- src/api/routers/pipeline.py + src/api/schemas.py — API contract for analysis requests.
- app/nicegui_dashboard/main.py + app/nicegui_dashboard/services/nicegui_api_client.py — Dashboard entry + data fetching.
- launcher/main.py + launcher/process_manager.py — Desktop launcher logic.
- pyproject.toml — Dependencies, optional groups (cli, api, dashboard-nicegui, diarize), scripts.
- configs/ — llm_config.yaml, alerting_config.yaml, qa_scorecards/*.yaml, install_defaults.
- tests/ — Extensive; run with pytest. Many test_*.py mirroring src/.
- docs/LLM_AGENT_GUIDE.md — **Most important for agents**: architecture philosophy, patterns, what to do/not do, security rules.

## 5. How to Work With This Codebase
- **Setup**: pip install -e "[cli,api,dashboard-nicegui]" then sentimentanalys download-asr . For dev: requirements-dev.txt + pre-commit.
- **Run**:
  - CLI: sentimentanalys --help or python -m src.cli
  - API: uvicorn src.api:app --reload (or via launcher)
  - Dashboard: python -m app.nicegui_dashboard.main
  - Windows: .\launcher.ps1 or Sentimentanalys.bat
  - Tests: pytest (or specific pytest tests/test_pipeline.py -q )
  - Evaluate: python -m src.evaluate fas4-validation
- **Coding Conventions**: Strict Pydantic models for all I/O and LLM output. Type hints everywhere. Logging via standard. Error handling with custom exceptions in src/core/errors.py. Swedish variable/docstrings where domain-specific. Pre-commit hooks (ruff, mypy?).
- **Adding Features**:
  - New analysis step: Implement in src/analysis/, register in registry.py, add to pipeline, update schemas/tests.
  - New LLM provider: Add client + analyzer in src/llm/, update LLM_PROVIDERS.md and schemas.
  - Dashboard tab/component: New file in components/, wire in main/layout, add to api_client if backend.
  - API endpoint: Router in routers/, service if needed, schema, test.
- **Testing**: Unit + integration. Mock external (LLM, ASR heavy). Use fixtures. Aim high coverage on src/.
- **Docs**: Update ROADMAP/CHANGELOG on releases. Use this AGENT_CONTEXT + PROJECT_STATUS as single source. Re-run github-project-status skill after significant changes.

## 6. Open Tasks & Priorities
- High-priority: v0.5 release + tagging + customer demo.
- Full production fine-tuning training loop on real callcenter data.
- Further real-time optimizations and Edge AI expansion.
- Sync ROADMAP.md with latest TASK numbering.

## 7. Context for Future Agents
After every change that affects features, re-run the github-project-status skill. We use Swedish/Norwegian localization in UI and prompts. Strong focus on PII/GDPR and graceful degradation. When implementing new analyzer or LLM feature, see src/analysis/registry.py and src/llm/schemas.py. Dashboard components should use nicegui_api_client.py for backend data.