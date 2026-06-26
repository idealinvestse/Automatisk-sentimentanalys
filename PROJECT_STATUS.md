# Project Status — Automatisk-sentimentanalys

**Last Updated:** 2026-06-26 (slut på session) via github-project-status skill  
**Repository:** https://github.com/idealinvestse/Automatisk-sentimentanalys  
**Current Branch:** main  
**Working Tree:** Clean

## Recent Activity
- **2026-06-26 (full day session)**: 
  - TASK-01 till TASK-06 slutförda (LLM-judge wiring + panel, alerting polish, code review, tester, circuit breaker status).
  - Ny testfil `tests/test_llm_judge_panel.py` med god täckning på filter-logik.
  - Ny API-router `src/api/routers/alerting.py` med `/status` och `/reset-circuit-breaker`.
  - Dashboard auto-hämtar nu alerting status.
  - Ny task TASK-07 tillagd (multi-worker robustness för circuit breaker).

## System Description
Automatisk-sentimentanalys är ett svenskt Call Center Intelligence-system för automatisk analys av kundtjänstsamtal. Systemet hanterar hela kedjan från ljudfil till insikter: ASR/transkribering (faster-whisper + WhisperX med diarization för svenska), multi-dimensionell analys (sentiment, emotion, intent-klassificering, aspect-based sentiment, topics/hot topics, insights/root cause), LLM-stöd (Mistral, OpenRouter, Groq med GDPR/EU-residency), PII-redaktion, compliance/QA-scorecards (regel + LLM hybrid), agent performance metrics, semantic search över samtal, alerting och realtids-dashboard.

Det löser problemet med manuell genomgång av samtal i call centers genom att automatisera transkription, sentiment- och intent-analys, QA, coaching-insikter och trenddetektion på svenska. Målgrupp är svenska call center-operatörer, QA-team och chefer som vill ha GDPR-vänlig, skalbar, self-hosted eller VPS-baserad lösning med Windows-desktop launcher för enkelhet.

Byggt modulärt i Python med registry-pattern för analys-steg, pluggable LLM-providers, FastAPI-backend, NiceGUI modern dashboard, CLI och PowerShell-launcher/installer. Starkt fokus på svenska språket, PII-skydd, caching/pre-computation för prestanda och omfattande tester/evaluering.

## Architecture Overview
- **Core Pipeline** (`src/pipeline.py`, `src/analysis/registry.py`): CallAnalysisPipeline som kör transkription → analysis steps (sentiment, emotion, intent, llm_judge, pii_redactor, etc.) → aggregation/insights. Graceful degradation och caching.
- **Transcription Layer** (`src/transcription/`): Factory för faster-whisper, WhisperX, Transformers; preprocess, diarization (`src/diarization.py`).
- **Analysis Modules** (`src/analysis/`, `src/sentiment.py`, `src/intent.py`, `src/llm/`): Sentiment, emotion, intent, aspect, topics, trajectory, llm_judge, role_classifier, pii_redactor, negation handling, blending.
- **LLM Providers** (`src/llm/`): GroqClient/Analyzer, MistralAnalyzer, OpenRouterClient; schemas, prompts, PII-safe routing. Fallback chains och EU-residency gate.
- **API Layer** (`src/api/`): FastAPI app med routers (transcription, pipeline, scan, text, conversation, ws_transcription, alerting). OpenAPI docs.
- **Dashboard** (`app/nicegui_dashboard/`): NiceGUI UI med tabs för overview, live analysis, transcription monitor, call detail, emotion timeline, hot topics wordcloud, insights, agent performance, qa_scorecard, alerts_panel (med dynamisk circuit breaker status), test_lab, pii_audit.
- **Launcher & Desktop** (`launcher/`, `Sentimentanalys.bat`, installer/): PowerShell launcher för ASR status/install/download/provision, UI panels, env builder. InnoSetup för Windows portable/exe.
- **Data & Training** (`data/`, `scripts/`, `src/finetune.py`, `src/evaluate.py`): Callcenter train/val datasets (CSV/JSONL), sensaldo lexicon, intent data; prepare, train_intent, evaluate_fas4_validation.
- **Infra** (`Dockerfile`, `docker-compose.nicegui.yml`, `pyproject.toml`, `Makefile`, `configs/`): Docker support, optional deps (cli, api, dashboard-nicegui, diarize), preflight/provision för modeller/secrets på Windows, YAML configs för llm, alerting, qa_scorecards.
- **Key Data Flows**: Audio upload → ASR → segments + speakers → pipeline analysis → structured report (sentiment scores, intents, QA, alerts, insights, llm_judge verdicts, alerting status) → stored/cached → dashboard/API consumption.

**Tech Stack**: Python 3.10+, PyTorch ecosystem (faster-whisper, pyannote for diarization), FastAPI + Uvicorn, NiceGUI, Pydantic, httpx, caching, pandas, OpenAI-compatible LLM clients, pytest (500+ tests), pre-commit, GitHub Actions CI.

**Deployment**: Local dev (pip install -e), Windows desktop (launcher.ps1 + portable), Docker for API/dashboard, VPS-ready with secrets handling. ASR models downloaded on demand.

## Feature Status

### Implemented / Live in Production
- Full ASR pipeline with Swedish-optimized models, diarization, preprocessing, batch & real-time WS transcription.
- Core analysis: sentiment (lexicon + ML), emotion, intent classification (trained), aspect/negation handling, PII redaction (hardened).
- LLM integration: Structured output via Pydantic, multiple providers (Mistral, OpenRouter, Groq with EU gate), caching, cost tracking.
- **LLM-judge**: Wired into pipeline + visualiserad i dashboard (`llm_judge_panel` med filter).
- Fas 4 Backend: agent_performance metrics + coaching, compliance_qa with YAML scorecards + evidence, insights_aggregator, semantic_search, alerting (rule-based + webhook with circuit breaker + status endpoint).
- REST API: Full coverage including new `/alerting/status` and `/alerting/reset-circuit-breaker`.
- **NiceGUI Dashboard**: Modern UI with KPI cards, calls table, live/transcription monitor, emotion timeline, wordcloud, insights, agent perf, QA scorecard, alerts_panel (with dynamic circuit breaker status), test_lab, pii_audit.
- CLI & Evaluation: Full CLI, evaluate fas4-validation, benchmarks, finetune scripts.
- Windows Experience: Launcher GUI/CLI for ASR management, provision, installer.
- Security/Compliance: PII hardening, GDPR considerations in LLM routing.
- Documentation: Extensive docs/, CHANGELOG, tests, PROJECT_STATUS.md, RECOMMENDED_NEXT_TASKS.md.

### In Progress / Partial
- Dynamic circuit breaker status in dashboard (backend + client ready, full auto-fetch in panel implemented).
- Dashboard performance optimizations (caching, async).

### Planned / Backlog
- TASK-07: Robust multi-worker circuit breaker state.
- Expanded finetuning and production model training.
- More LLM providers or local inference options.
- Production deployment automation and scaling.

## Known Issues & Technical Debt
- Circuit breaker state är per-process (modulnivå). Behöver förbättras för multi-worker deployment (TASK-07).
- LLM-judge data shape in some legacy paths may need small normalization.
- "Mark as handled" for alerts is currently session-only (in-memory).
- Old `llm_judge_breakdown.py` still exists (can be deprecated).

## Recommendations for Next Development Sessions
- **High priority**: TASK-07 – Gör circuit breaker state robust för multi-worker deployment.
- Kör full test suite (inklusive nya `test_llm_judge_panel.py`).
- Validera end-to-end flöde för LLM-judge + alerting status.
- Re-run `github-project-status` och `github-repo-deep-dive` efter nästa kodningsrunda.

**Current recommended focus**:
1. TASK-07 (multi-worker robustness).
2. Full test coverage + CI validation.
3. Production readiness för alerting och LLM-judge.