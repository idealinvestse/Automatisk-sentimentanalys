# Project Status — Automatisk-sentimentanalys

**Last Updated:** 2026-06-26 via github-project-status skill (after LLM-judge + alerting polish session)  
**Repository:** https://github.com/idealinvestse/Automatisk-sentimentanalys  
**Current Branch:** main  
**Working Tree:** Clean

## Recent Activity
- **2026-06-26 (today)**: 
  - **TASK-01**: Wired `LLMJudgeAnalyzer` into `CallAnalysisPipeline` via `_build_analyzer_configs()` (foundational integration).
  - **TASK-02**: Created new `llm_judge_panel.py` component (card-based with filter for changed verdicts) and integrated it into `call_detail.py`.
  - **Alerting polish (TASK-04)**: Improved `alerts_panel.py` with separate "Hanterade alerts" section, better mark-as-handled UX, and webhook/circuit breaker status indicator.
  - Previous: Groq integration, Fas 4 backend, PII hardening, NiceGUI dashboard enhancements.

## System Description
Automatisk-sentimentanalys är ett svenskt Call Center Intelligence-system för automatisk analys av kundtjänstsamtal. Systemet hanterar hela kedjan från ljudfil till insikter: ASR/transkribering (faster-whisper + WhisperX med diarization för svenska), multi-dimensionell analys (sentiment, emotion, intent-klassificering, aspect-based sentiment, topics/hot topics, insights/root cause), LLM-stöd (Mistral, OpenRouter, Groq med GDPR/EU-residency), PII-redaktion, compliance/QA-scorecards (regel + LLM hybrid), agent performance metrics, semantic search över samtal, alerting och realtids-dashboard.

Det löser problemet med manuell genomgång av samtal i call centers genom att automatisera transkription, sentiment- och intent-analys, QA, coaching-insikter och trenddetektion på svenska. Målgrupp är svenska call center-operatörer, QA-team och chefer som vill ha GDPR-vänlig, skalbar, self-hosted eller VPS-baserad lösning med Windows-desktop launcher för enkelhet.

Byggt modulärt i Python med registry-pattern för analys-steg, pluggable LLM-providers, FastAPI-backend, NiceGUI modern dashboard, CLI och PowerShell-launcher/installer. Starkt fokus på svenska språket, PII-skydd, caching/pre-computation för prestanda och omfattande tester/evaluering.

## Architecture Overview
- **Core Pipeline** (`src/pipeline.py`, `src/analysis/registry.py`): CallAnalysisPipeline som kör transkription → analysis steps (sentiment, emotion, intent, llm_judge, pii_redactor, etc.) → aggregation/insights. Graceful degradation och caching.
- **Transcription Layer** (`src/transcription/`): Factory för faster-whisper, WhisperX, Transformers; preprocess, diarization (`src/diarization.py`).
- **Analysis Modules** (`src/analysis/`, `src/sentiment.py`, `src/intent.py`, `src/llm/`): Sentiment, emotion, intent, aspect, topics, trajectory, llm_judge, role_classifier, pii_redactor, negation handling, blending.
- **LLM Providers** (`src/llm/`): GroqClient/Analyzer, MistralAnalyzer, OpenRouterClient; schemas, prompts, PII-safe routing. Fallback chains och EU-residency gate.
- **API Layer** (`src/api/`): FastAPI app med routers (transcription, pipeline, scan, text, conversation, ws_transcription), schemas, middleware (rate limit), services, batch jobs. OpenAPI docs.
- **Dashboard** (`app/nicegui_dashboard/`): NiceGUI UI med tabs för overview, live analysis, transcription monitor, call detail, emotion timeline, hot topics wordcloud, insights, agent performance, qa_scorecard, alerts_panel, test_lab, pii_audit. Använder NiceGUIAPIClient mot backend eller local demo/fas4_data. Ny `llm_judge_panel` integrerad i call_detail.
- **Launcher & Desktop** (`launcher/`, `Sentimentanalys.bat`, installer/): PowerShell launcher för ASR status/install/download/provision, UI panels, env builder. InnoSetup för Windows portable/exe.
- **Data & Training** (`data/`, `scripts/`, `src/finetune.py`, `src/evaluate.py`): Callcenter train/val datasets (CSV/JSONL), sensaldo lexicon, intent data; prepare, train_intent, evaluate_fas4_validation.
- **Infra** (`Dockerfile`, `docker-compose.nicegui.yml`, `pyproject.toml`, `Makefile`, `configs/`): Docker support, optional deps (cli, api, dashboard-nicegui, diarize), preflight/provision för modeller/secrets på Windows, YAML configs för llm, alerting, qa_scorecards.
- **Key Data Flows**: Audio upload → ASR → segments + speakers → pipeline analysis (parallel where possible) → structured report (sentiment scores, intents, QA, alerts, insights, llm_judge verdicts) → stored/cached → dashboard/API consumption.

**Tech Stack**: Python 3.10+, PyTorch ecosystem (faster-whisper, pyannote for diarization), FastAPI + Uvicorn, NiceGUI, Pydantic, httpx, caching, pandas, OpenAI-compatible LLM clients, pytest (500+ tests), pre-commit, GitHub Actions CI.

**Deployment**: Local dev (pip install -e), Windows desktop (launcher.ps1 + portable), Docker for API/dashboard, VPS-ready with secrets handling. ASR models downloaded on demand.

## Feature Status

### Implemented / Live in Production
- Full ASR pipeline with Swedish-optimized models, diarization, preprocessing, batch & real-time WS transcription.
- Core analysis: sentiment (lexicon + ML), emotion, intent classification (trained), aspect/negation handling, PII redaction (hardened).
- LLM integration: Structured output via Pydantic, multiple providers (Mistral, OpenRouter, Groq with EU gate), caching, cost tracking.
- **LLM-judge** (new): Wired into pipeline (TASK-01). Runs on low-confidence sentiment segments with batching, budget control and graceful fallback.
- Fas 4 Backend: agent_performance metrics + coaching, compliance_qa with YAML scorecards + evidence, insights_aggregator (hot topics, trends), semantic_search (hybrid), alerting (rule-based + webhook with circuit breaker).
- REST API: Full coverage for transcription, pipeline, batch, semantic search, qa/score, alerts, agent_performance.
- **NiceGUI Dashboard**: Modern UI with KPI cards, calls table, live/transcription monitor, emotion timeline, wordcloud, insights, agent perf, QA scorecard, alerts_panel (improved handled section), test_lab, pii_audit. New `llm_judge_panel` integrated in call_detail.
- CLI & Evaluation: Full CLI, evaluate fas4-validation, benchmarks, finetune scripts.
- Windows Experience: Launcher GUI/CLI for ASR management, provision, installer.
- Security/Compliance: PII hardening, GDPR considerations in LLM routing.
- Documentation: Extensive docs/, CHANGELOG, tests.

### In Progress / Partial
- Full production use of LLM-judge verdicts in dashboard and API responses (panel is live, data flow improving).
- Alerting webhook testability and "mark as handled" persistence.
- Dashboard performance optimizations (caching, async).

### Planned / Backlog
- Expanded finetuning and production model training.
- More LLM providers or local inference options.
- Production deployment automation and scaling.
- Deferred items from ROADMAP (YouTube ingest etc.).

## Known Issues & Technical Debt
- LLM-judge data shape in some legacy paths may need small normalization.
- "Mark as handled" for alerts is currently session-only (in-memory).
- Some Fas4 KPI stubs in evaluate.py still present.
- High test count but integration tests for new LLM-judge paths recommended.

## Recommendations for Next Development Sessions
- **High priority**: Test LLM-judge end-to-end (pipeline → API → dashboard panel) with real low-confidence segments.
- Polish alerting further (test webhook button in test_lab, better persistence for handled alerts).
- Improve dashboard caching and async data loading for better perceived performance.
- Run full test suite after LLM-judge changes.
- Update `RECOMMENDED_NEXT_TASKS.md` and re-run this skill after next coding session.
- Always read `docs/LLM_AGENT_GUIDE.md`, `ROADMAP.md` and `SECURITY.md` before major changes.

**Next recommended focus areas** (from github-repo-deep-dive):
1. End-to-end validation of LLM-judge feature.
2. Alerting testability and UX polish.
3. Dashboard reliability (caching/async).