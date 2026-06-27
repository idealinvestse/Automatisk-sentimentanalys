# Project Status — Automatisk-sentimentanalys

**Last Updated:** 2026-06-27 14:45 CEST via github-project-status skill  
**Repository:** https://github.com/idealinvestse/Automatisk-sentimentanalys  
**Current Branch:** main  
**Working Tree:** Clean

## Recent Activity
- Sprint "Production & Advanced Capabilities" (TASK-14 to TASK-18) completed and pushed
- Full fine-tuning pipeline + live training + model selector in dashboard
- Production real-time (WebSocket), multi-worker scaling, Docker/VPS ready
- Advanced automation (auto-reports, scheduled insights, external API)
- UX/accessibility polish, multi-user, demo-mode
- Full tests, security audit, v0.5 prep

## System Description
Automatisk-sentimentanalys är ett svenskt Call Center Intelligence-system för automatisk analys av kundtjänstsamtal. Systemet hanterar hela kedjan från ljudfil till insikter: ASR/transkribering (faster-whisper + WhisperX med diarization för svenska), multi-dimensionell analys (sentiment, emotion, intent-klassificering, aspect-based sentiment, topics/hot topics, insights/root cause), LLM-stöd (Mistral, OpenRouter, Groq med GDPR/EU-residency), PII-redaktion, compliance/QA-scorecards (regel + LLM hybrid), agent performance metrics, semantic search över samtal, alerting och realtids-dashboard.

Det löser problemet med manuell genomgång av samtal i call centers genom att automatisera transkription, sentiment- och intent-analys, QA, coaching-insikter och trenddetektion på svenska. Målgrupp är svenska call center-operatörer, QA-team och chefer som vill ha GDPR-vänlig, skalbar, self-hosted eller VPS-baserad lösning med Windows-desktop launcher för enkelhet.

Byggt modulärt i Python med registry-pattern för analys-steg, pluggable LLM-providers, FastAPI-backend, NiceGUI modern dashboard, CLI och PowerShell-launcher/installer. Starkt fokus på svenska språket, PII-skydd, caching/pre-computation för prestanda och omfattande tester/evaluering.

## Architecture Overview
- **Core Pipeline** (`src/pipeline.py`, `src/analysis/registry.py`): CallAnalysisPipeline som kör transkription 	o analysis steps (sentiment, emotion, intent, llm_judge, pii_redactor, etc.) 	o aggregation/insights. Graceful degradation och caching.
- **Transcription Layer** (`src/transcription/`): Factory för faster-whisper, WhisperX, Transformers; preprocess, diarization (`src/diarization.py`).
- **Analysis Modules** (`src/analysis/`, `src/sentiment.py`, `src/intent.py`, `src/llm/`): Sentiment, emotion, intent, aspect, topics, trajectory, llm_judge, role_classifier, pii_redactor, negation handling, blending.
- **LLM Providers** (`src/llm/`): GroqClient/Analyzer, MistralAnalyzer, OpenRouterClient; schemas, prompts, PII-safe routing. Fallback chains och EU-residency gate.
- **API Layer** (`src/api/`): FastAPI app med routers (transcription, pipeline, scan, text, conversation, ws_transcription, alerting). OpenAPI docs.
- **Dashboard** (`app/nicegui_dashboard/`): NiceGUI UI med tabs för overview, live analysis, transcription monitor, call detail, emotion timeline, hot topics wordcloud, insights, agent performance, qa_scorecard, alerts_panel, test_lab, pii_audit. Många komponenter (advanced_insights, analytics_trends, etc.).
- **Launcher & Desktop** (`launcher/`, `Sentimentanalys.bat`, installer/): PowerShell launcher för ASR status/install/download/provision, UI panels, env builder. InnoSetup för Windows portable/exe.
- **Data & Training** (`data/`, `scripts/`, `src/finetune.py`, `src/evaluate.py`): Callcenter train/val datasets (CSV/JSONL), sensaldo lexicon, intent data; prepare, train_intent, evaluate_fas4_validation. Fine_tuning dir.
- **Infra** (`Dockerfile`, `docker-compose.nicegui.yml`, `pyproject.toml`, `Makefile`, `configs/`): Docker support, optional deps (cli, api, dashboard-nicegui, diarize), preflight/provision för modeller/secrets på Windows, YAML configs för llm, alerting, qa_scorecards.
- **Key Data Flows**: Audio upload 	o ASR 	o segments + speakers 	o pipeline analysis 	o structured report (sentiment scores, intents, QA, alerts, insights, llm_judge verdicts, alerting status) 	o stored/cached 	o dashboard/API consumption.

**Tech Stack**: Python 3.10+, PyTorch ecosystem (faster-whisper, pyannote for diarization), FastAPI + Uvicorn, NiceGUI, Pydantic, httpx, caching, pandas, OpenAI-compatible LLM clients, pytest (500+ tests), pre-commit, GitHub Actions CI.

**Deployment**: Local dev (pip install -e), Windows desktop (launcher.ps1 + portable), Docker for API/dashboard, VPS-ready with secrets handling. ASR models downloaded on demand.

## Feature Status

### Implemented / Live in Production
- Full ASR pipeline with Swedish-optimized models, diarization, preprocessing, batch & real-time WS transcription.
- Core analysis: sentiment (lexicon + ML), emotion, intent classification (trained), aspect/negation handling, PII redaction (hardened).
- LLM integration: Structured output via Pydantic, multiple providers (Mistral, OpenRouter, Groq with EU gate), caching, cost tracking.
- LLM-judge wired into pipeline + visualized in dashboard.
- Fas 4 Backend: agent_performance metrics + coaching, compliance_qa with YAML scorecards + evidence, insights_aggregator, semantic_search, alerting (rule-based + webhook with circuit breaker + status endpoint).
- REST API: Full coverage including new /alerting/status and /alerting/reset-circuit-breaker.
- NiceGUI Dashboard: Modern UI with KPI cards, calls table, live/transcription monitor, emotion timeline, wordcloud, insights, agent perf, QA scorecard, alerts_panel, test_lab, pii_audit. Advanced components (advanced_insights, analytics_trends, etc.).
- CLI & Evaluation: Full CLI, evaluate fas4-validation, benchmarks, finetune scripts.
- Windows Experience: Launcher GUI/CLI for ASR management, provision, installer.
- Security/Compliance: PII hardening, GDPR considerations in LLM routing.
- Documentation: Extensive docs/, CHANGELOG, tests, PROJECT_STATUS.md, RECOMMENDED_NEXT_TASKS.md.
- Fine-tuning: Directory + basic pipeline + model selector in dashboard (TASK-14 completed).
- Real-time & Scaling: WebSocket, multi-worker alerting state (JSON persistent), Docker/VPS ready (TASK-15).
- Automation: Auto-reports, scheduled insights, external API (TASK-16).
- UX: Polish, accessibility, multi-user, demo-mode (TASK-17).

### In Progress / Partial
- Full production fine-tuning training loop and evaluation on real callcenter data.
- v0.5 release preparation and tagging.

### Planned / Backlog
- Voice Synthesis / advanced multi-modal features.
- Edge AI Network expansion.
- Customer onboarding and DK market launch integration.
- Further optimization of LLM costs and latency.

## Known Issues & Technical Debt
- Some legacy llm_judge_breakdown.py still exists (can be deprecated).
- Fine-tuning full training scripts need more real anonymized data.
- Documentation in ROADMAP.md needs sync with latest TASK numbering.

## Recommendations for Next Development Sessions
- High priority: v0.5 release + tagging + demo.
- Kör full test suite + E2E validation.
- Re-run github-project-status and github-repo-deep-dive after next coding round.
- Focus on production deployment and customer demo.