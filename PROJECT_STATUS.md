# Project Status — Automatisk-sentimentanalys

**Last Updated:** 2026-06-26 via github-project-status skill  
**Repository:** https://github.com/idealinvestse/Automatisk-sentimentanalys  
**Current Branch:** main @ 155a75ccc45d7ffaab2f9b1cd2cd8a68805ac4eb  
**Working Tree:** Clean (remote analysis via GitHub API)

## Recent Activity
- **2026-06-24**: Groq Cloud LLM integration added (new provider, EU residency gate, caching, cost tracking). Fas 3/4 NiceGUI dashboard enhancements (Agent Performance, Fas4 Insikter tabs, alerts, PII audit). PII-redaction hardening and alerting webhook production-grade improvements. Doc reconciliation.
- **Unreleased**: Focus on LLM-judge completion, dashboard visualization polish, production readiness for Fas 4 backend features.
- Last significant commits around Fas4 backend completion, 509+ tests, high coverage on src/.
- No uncommitted changes detected in this scan.

## System Description
Automatisk-sentimentanalys är ett svenskt Call Center Intelligence-system för automatisk analys av kundtjänstsamtal. Systemet hanterar hela kedjan från ljudfil till insikter: ASR/transkribering (faster-whisper + WhisperX med diarization för svenska), multi-dimensionell analys (sentiment, emotion, intent-klassificering, aspect-based sentiment, topics/hot topics, insights/root cause), LLM-stöd (Mistral, OpenRouter, nyligen Groq med GDPR/EU-residency), PII-redaktion, compliance/QA-scorecards (regel + LLM hybrid), agent performance metrics, semantic search över samtal, alerting och realtids-dashboard.

Det löser problemet med manuell genomgång av samtal i call centers genom att automatisera transkription, sentiment- och intent-analys, QA, coaching-insikter och trenddetektion på svenska. Målgrupp är svenska call center-operatörer, QA-team och chefer som vill ha GDPR-vänlig, skalbar, self-hosted eller VPS-baserad lösning med Windows-desktop launcher för enkelhet.

Byggt modulärt i Python med registry-pattern för analys-steg, pluggable LLM-providers, FastAPI-backend, NiceGUI modern dashboard, CLI och PowerShell-launcher/installer. Starkt fokus på svenska språket, PII-skydd, caching/pre-computation för prestanda och omfattande tester/evaluering.

## Architecture Overview
- **Core Pipeline** (`src/pipeline.py`, `src/analysis/registry.py`): CallAnalysisPipeline som kör transkription → analysis steps (sentiment, emotion, intent, llm_judge, pii_redactor, etc.) → aggregation/insights. Graceful degradation och caching.
- **Transcription Layer** (`src/transcription/`): Factory för faster-whisper, WhisperX, Transformers; preprocess, diarization (`src/diarization.py`).
- **Analysis Modules** (`src/analysis/`, `src/sentiment.py`, `src/intent.py`, `src/llm/`): Sentiment, emotion, intent, aspect, topics, trajectory, llm_judge, role_classifier, pii_redactor, negation handling, blending.
- **LLM Providers** (`src/llm/`): GroqClient/Analyzer (ny), MistralAnalyzer, OpenRouterClient; schemas, prompts, PII-safe routing. Fallback chains och EU-residency gate.
- **API Layer** (`src/api/`): FastAPI app med routers (transcription, pipeline, scan, text, conversation, ws_transcription), schemas, middleware (rate limit), services, batch jobs. OpenAPI docs.
- **Dashboard** (`app/nicegui_dashboard/`): NiceGUI UI med tabs för overview, live analysis, transcription monitor, call detail, emotion timeline, hot topics wordcloud, insights, agent performance, qa_scorecard, alerts_panel, test_lab, pii_audit. Använder NiceGUIAPIClient mot backend eller local demo/fas4_data.
- **Launcher & Desktop** (`launcher/`, `Sentimentanalys.bat`, installer/): PowerShell launcher för ASR status/install/download/provision, UI panels, env builder. InnoSetup för Windows portable/exe.
- **Data & Training** (`data/`, `scripts/`, `src/finetune.py`, `src/evaluate.py`): Callcenter train/val datasets (CSV/JSONL), sensaldo lexicon, intent data; prepare, train_intent, evaluate_fas4_validation.
- **Infra** (`Dockerfile`, `docker-compose.nicegui.yml`, `pyproject.toml`, `Makefile`, `configs/`): Docker support, optional deps (cli, api, dashboard-nicegui, diarize), preflight/provision för modeller/secrets på Windows, YAML configs för llm, alerting, qa_scorecards.
- **Key Data Flows**: Audio upload → ASR → segments + speakers → pipeline analysis (parallel where possible) → structured report (sentiment scores, intents, QA, alerts, insights) → stored/cached → dashboard/API consumption. Semantic search and aggregations for trends.

**Tech Stack**: Python 3.10+, PyTorch ecosystem (faster-whisper, pyannote for diarization), FastAPI + Uvicorn, NiceGUI, Pydantic, httpx, SQLAlchemy? (caching), pandas, scikit-learn?, OpenAI-compatible LLM clients, pytest (509+ tests), pre-commit, GitHub Actions CI.

**Deployment**: Local dev (pip install -e), Windows desktop (launcher.ps1 + portable), Docker for API/dashboard, VPS-ready with secrets handling. ASR models downloaded on demand (kb-whisper etc.).

## Feature Status

### Implemented / Live in Production
- Full ASR pipeline with Swedish-optimized models, diarization, preprocessing, batch & real-time WS transcription.
- Core analysis: sentiment (lexicon + ML), emotion, intent classification (trained), aspect/negation handling, PII redaction (hardened with Luhn, Swedish names/addresses/phones).
- LLM integration: Structured output via Pydantic, multiple providers (Mistral, OpenRouter, Groq with EU gate), caching, cost tracking, prompts for judge/QA/summary.
- Fas 4 Backend: agent_performance metrics + coaching, compliance_qa with YAML scorecards + evidence, insights_aggregator (hot topics, trends), semantic_search (hybrid), alerting (rule-based + webhook with circuit breaker).
- REST API: Full coverage for transcription, pipeline, batch, semantic search, qa/score, alerts, agent_performance; rate limiting, error contracts, OpenAPI.
- NiceGUI Dashboard: Modern UI with KPI cards, calls table, live/transcription monitor, emotion timeline, wordcloud, insights, agent perf, QA scorecard, alerts, test lab, PII audit, onboarding. Local demo + API client modes.
- CLI & Evaluation: Full CLI (sentimentanalys), evaluate fas4-validation, benchmarks for audio models, finetune scripts.
- Windows Experience: Launcher GUI/CLI for ASR management, provision, installer (Inno), portable build.
- Data & Models: Prepared callcenter datasets, sensaldo lexicon, intent training, reports from validation.
- Security/Compliance: PII hardening, GDPR considerations in LLM routing, SECURITY.md, preflight checks.
- Documentation: Extensive docs/ (ARCHITECTURE, LLM_AGENT_GUIDE, FAS4_COMPLETION, API, ROADMAP, plans), CHANGELOG, tests.

### In Progress / Partial
- LLM-judge full integration and dashboard visualization polish (Fas 4 visualisering).
- Alerting webhook production use and mark-as-handled flows.
- YouTube ingest rollback / deferred items noted in docs.
- Some stubs in llm_judge and edge cases in Fas4 evaluation.
- Dashboard refresh optimizations and async handling refinements (recent fixes applied).

### Planned / Backlog
- Data/finetuning expansion and production model training.
- Full production deployment guides, scaling (Redis cache?).
- Additional languages? or deeper Swedish domain adaptation.
- YouTube/long-form ingest if revived.
- Enhanced real-time coaching / live alerts in dashboard.
- More LLM providers or self-hosted options.
- Mobile/responsive improvements or embedded widgets for call centers.

## Known Issues & Technical Debt
- LLM-judge is partially stubbed in some paths (noted in ROADMAP).
- Alerting webhook has TODO for full production webhook config/UI.
- Some Fas4 KPI stubs in evaluate.py.
- High test count but some integration tests may need hardware (GPU for ASR/diarize).
- Dependency on external LLM APIs (cost, latency, GDPR); local fallback limited.
- Windows-specific launcher strong, but cross-platform dashboard/API primary.
- Documentation debt low thanks to recent reconciliation, but keep AGENT_CONTEXT synced.

## Recommendations for Next Development Sessions
- Prioritize completing LLM-judge integration and wiring into dashboard + API for full Fas4 value.
- Run full test suite + `python -m src.evaluate fas4-validation` after changes affecting pipeline.
- When editing analysis/llm components: see `src/analysis/registry.py`, `src/llm/schemas.py` and `docs/LLM_AGENT_GUIDE.md` for patterns (strict schemas, graceful degradation, PII-safe).
- For dashboard: use `app/nicegui_dashboard/services/nicegui_api_client.py` and `fas4_data.py` for data layer; follow NiceGUI refreshable + async patterns from recent fixes.
- Update PROJECT_STATUS.md and AGENT_CONTEXT.md after every major session using this skill.
- Next: Polish Fas4 insights/hot topics in UI, add webhook test in test_lab, prepare v0.5.0 release notes.
- Always read `docs/LLM_AGENT_GUIDE.md`, `ROADMAP.md` and `SECURITY.md` before code changes.