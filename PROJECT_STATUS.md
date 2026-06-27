# Project Status — Automatisk-sentimentanalys

**Last Updated:** 2026-06-27 15:05 CEST via github-project-status skill  
**Repository:** https://github.com/idealinvestse/Automatisk-sentimentanalys  
**Current Branch:** main  
**Working Tree:** Clean (recent pushes for model catalog, CLI command, dashboard button, dynamic pricing)

## Recent Activity
- Configurable storage path (disk/mapp) for LLM models & downloads selectable in Launcher + Dashboard
- **OpenRouter Model Catalog Scanner** (`src/llm/model_catalog.py`): Full scan of ~200+ models with id, name, short description, context_length, pricing (per token + per million USD)
- **Typer CLI command**: `sentimentanalys scan-openrouter-models` with rich table of cheapest models
- **NiceGUI Dashboard button** in Test Lab: "Scanna OpenRouter modeller + uppdatera pricing" + status + quick view of top models
- **Dynamic PRICING** in `openrouter_client.py`: `refresh_pricing_from_catalog()` + auto-use in `_compute_approx_cost()` for live cost tracking
- Updated `test_lab.py` with LLM model settings card
- All changes pushed and integrated

## System Description
Automatisk-sentimentanalys är ett svenskt Call Center Intelligence-system för automatisk analys av kundtjänstsamtal. Systemet hanterar hela kedjan från ljudfil till insikter: ASR/transkribering (faster-whisper + WhisperX med diarization för svenska), multi-dimensionell analys (sentiment, emotion, intent-klassificering, aspect-based sentiment, topics/hot topics, insights/root cause), LLM-stöd (Mistral, OpenRouter, Groq med GDPR/EU-residency), PII-redaktion, compliance/QA-scorecards (regel + LLM hybrid), agent performance metrics, semantic search över samtal, alerting och realtids-dashboard.

Det löser problemet med manuell genomgång av samtal i call centers genom att automatisera transkription, sentiment- och intent-analys, QA, coaching-insikter och trenddetektion på svenska. Målgrupp är svenska call center-operatörer, QA-team och chefer som vill ha GDPR-vänlig, skalbar, self-hosted eller VPS-baserad lösning med Windows-desktop launcher för enkelhet.

Byggt modulärt i Python med registry-pattern för analys-steg, pluggable LLM-providers, FastAPI-backend, NiceGUI modern dashboard, CLI och PowerShell-launcher/installer. Starkt fokus på svenska språket, PII-skydd, caching/pre-computation för prestanda och omfattande tester/evaluering.

## Architecture Overview
- **Core Pipeline** (`src/pipeline.py`, `src/analysis/registry.py`): CallAnalysisPipeline som kör transkription → analysis steps (sentiment, emotion, intent, llm_judge, pii_redactor, etc.) → aggregation/insights. Graceful degradation och caching.
- **Transcription Layer** (`src/transcription/`): Factory för faster-whisper, WhisperX, Transformers; preprocess, diarization (`src/diarization.py`).
- **Analysis Modules** (`src/analysis/`, `src/sentiment.py`, `src/intent.py`, `src/llm/`): Sentiment, emotion, intent, aspect, topics, trajectory, llm_judge, role_classifier, pii_redactor, negation handling, blending.
- **LLM Providers** (`src/llm/`): GroqClient/Analyzer, MistralAnalyzer, OpenRouterClient; schemas, prompts, PII-safe routing. Fallback chains och EU-residency gate. **Ny**: Dynamic pricing från model_catalog.
- **API Layer** (`src/api/`): FastAPI app med routers (transcription, pipeline, scan, text, conversation, ws_transcription, alerting). OpenAPI docs.
- **Dashboard** (`app/nicegui_dashboard/`): NiceGUI UI med tabs för overview, live analysis, transcription monitor, call detail, emotion timeline, hot topics wordcloud, insights, agent perf, qa_scorecard, alerts_panel, test_lab (nu med LLM Model Catalog knapp), pii_audit. Många komponenter.
- **Launcher & Desktop** (`launcher/`): PowerShell launcher för ASR status/install/download/provision, UI panels, env builder. InnoSetup för Windows portable/exe. **Ny**: Valbar mapp för modeller.
- **Data & Training** (`data/`, `scripts/`, `src/finetune.py`): Callcenter train/val datasets, sensaldo lexicon, intent data; fine_tuning dir.
- **Infra** (`Dockerfile`, `docker-compose.nicegui.yml`, `pyproject.toml`, `Makefile`, `configs/`): Docker support, optional deps, configs (llm_config.yaml m.m.).
- **Key Data Flows**: Audio upload → ASR → segments + speakers → pipeline analysis → structured report → stored/cached → dashboard/API consumption. LLM calls nu med live pricing från catalog.
- **Grok Build Optimization**: `.grok/skills/` med 6 custom skills + AGENTS.md + Grok Build quickstart.

**Tech Stack**: Python 3.10+, PyTorch ecosystem (faster-whisper, pyannote), FastAPI + Uvicorn, NiceGUI, Pydantic, openai SDK, httpx/urllib, caching, pandas, pytest (500+ tests), pre-commit, GitHub Actions CI.

**Deployment**: Local dev (pip install -e), Windows desktop (launcher.ps1 + portable), Docker for API/dashboard, VPS-ready with secrets handling. ASR + LLM models downloaded on demand to configurable path.

## Feature Status

### Implemented / Live in Production
- Full ASR pipeline with Swedish-optimized models, diarization, preprocessing, batch & real-time WS transcription.
- Core analysis: sentiment (lexicon + ML), emotion, intent classification (trained), aspect/negation handling, PII redaction (hardened).
- LLM integration: Structured output via Pydantic, multiple providers (Mistral, OpenRouter, Groq with EU gate), caching, cost tracking. **Ny**: Dynamic pricing from OpenRouter catalog.
- LLM-judge wired into pipeline + visualized in dashboard.
- Fas 4 Backend: agent_performance metrics + coaching, compliance_qa with YAML scorecards + evidence, insights_aggregator, semantic_search, alerting (rule-based + webhook with circuit breaker + status endpoint).
- REST API: Full coverage including new /alerting/status and /alerting/reset-circuit-breaker.
- NiceGUI Dashboard: Modern UI with KPI cards, calls table, live/transcription monitor, emotion timeline, wordcloud, insights, agent perf, QA scorecard, alerts_panel, test_lab (med LLM catalog refresh knapp), pii_audit. Advanced components.
- CLI & Evaluation: Full CLI, evaluate fas4-validation, benchmarks, finetune scripts. **Ny**: `scan-openrouter-models` kommando.
- Windows Experience: Launcher GUI/CLI for ASR management, provision, installer. **Ny**: Valbar download-mapp.
- Security/Compliance: PII hardening, GDPR considerations in LLM routing.
- Documentation: Extensive docs/, CHANGELOG, tests, PROJECT_STATUS.md, AGENT_CONTEXT.md, .grok/skills/.
- Fine-tuning: Directory + basic pipeline + model selector in dashboard.
- Real-time & Scaling: WebSocket, multi-worker alerting state (JSON persistent), Docker/VPS ready.
- Automation: Auto-reports, scheduled insights, external API.
- **LLM Model Management**: Full OpenRouter catalog scanner, dynamic pricing, CLI + Dashboard integration.

### In Progress / Partial
- Full production fine-tuning training loop and evaluation on real callcenter data.
- v0.5 release preparation and tagging.

### Planned / Backlog
- Voice Synthesis / advanced multi-modal features.
- Edge AI Network expansion.
- Customer onboarding and DK market launch integration.
- Further optimization of LLM costs and latency using catalog.

## Known Issues & Technical Debt
- Some legacy llm_judge_breakdown.py still exists (can be deprecated).
- Fine-tuning full training scripts need more real anonymized data.
- Documentation in ROADMAP.md needs sync with latest TASK numbering.
- model_catalog.json bör ignoreras i .gitignore för stora filer.

## Recommendations for Next Development Sessions
- High priority: v0.5 release + tagging + customer demo.
- Kör full test suite + E2E validation.
- Re-run github-project-status and github-repo-deep-dive after next coding round.
- Focus on production deployment and customer demo.
- Använd de nya skillsen i .grok/skills/ (code-review-reflector, repo-health-check m.fl.) för kvalitet och momentum.
- Nästa: Model picker i dashboard + auto cost-optimized model routing baserat på catalog.