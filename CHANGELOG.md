# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Unified observability (PROD-01)** — `StatusReporter` (`src/core/status.py`), context-aware logging (`get_logger`, `log_context`), `GET /status/processes`, `GET /status/health/detail`, script bootstrap (`scripts/_bootstrap.py`), launcher/API/CLI/dashboard integration.
- **Intent corpus** — deduplicated balanced train/val JSONL; honest macro F1 metrics in `reports/intent_baseline.json`.
- **Golden pipeline tests** — 6 callcenter scenarios + `@pytest.mark.slow` unmocked billing test.
- **API production guards** — `API_PRODUCTION`, `API_REQUIRE_AUTH`, `API_REQUIRE_MEDIA_ROOT` env vars.
- **Structured JSON logging** (`src/core/logging_config.py`) — `SENTIMENT_JSON_LOGS=1`.
- **Extended Prometheus metrics** — pipeline, analyzer, LLM, cache counters/histograms.
- **Edge AI MVP** — `sentimentanalys edge-analyze`, `src/edge/local_inference.py`, `docs/EDGE_AI.md`.
- **Model routing** (`src/llm/routing.py`) — FAST/BALANCED/DEEP tiers via `model_catalog`.
- **Fine-tuning CI** — `configs/finetune.ci.yaml`, smoke job, baseline eval gate.
- **GPU Docker** — `Dockerfile.gpu` (CUDA 12.1).
- **Domain corpus tools** — `scripts/validate_domain_corpus.py`, expanded `prepare_callcenter_data.py`.

### Changed
- **Intent heuristic** — phrase boosts, disambiguation rules; ~76% macro F1 on honest `intent_val.jsonl`.
- **Emotion analyzer** — removed broad `hur`/`vad` förvirring markers.
- **callcenter profile** — `dialect_sensitivity` disabled by default.
- **LLM quality eval** — schema pass rate + deep-path eligibility in `evaluate llm-quality`.
- **Deep path skip** — LLM-superseded analyzers skipped when `should_use_any_llm()` is true.
- **callcenter profile** — empathy/insights/trajectory moved to optional defaults.
- **Documentation v0.5 sync** — ROADMAP priorities, root plan archive, PROPOSED_ANALYZERS → ANALYZER_STRATEGY.
- **PRODUCTION_CHECKLIST** — checked implemented items; added prod env vars.

### Added (prior)
- **Production checklist** (`docs/PRODUCTION_CHECKLIST.md`) — observability, secrets, GPU Docker, metrics (DOC-02).
- **Prometheus `/metrics`** endpoint with alerting circuit-breaker gauges (OBS-01).

### Changed
- **DEPS-01**: Removed all `requirements*.txt`; install via `pip install -e ".[...]"` and `pyproject.toml` optional-deps only.
- **PIPE-01**: Extracted Fas-4 enrichment and LLM routing to `src/pipeline_steps.py`; `pipeline.py` under 550 LOC.
- **Documentation cleanup** (2026-06-28): `docs/CLEANUP_PLAN.md`, archive consolidation (`docs/archive/`), canonical `docs/ROADMAP.md`, stub root pointers, removed OpenClaw duplicates and agent build artifacts (`plan.md`, `memory/grok-plans/`).
- **Streamlit removed**: `app/setup_hub.py` deleted; launcher/config only supports NiceGUI dashboard.
- **Pipeline refactor**: shared `_run_local_analysis`, `_run_fas4_enrichment`, `_build_report` helpers (~720 lines in `pipeline.py`).
- **`predictive` analyzer** now delegates to `RiskAnalyzer` (numeric 0–1 scores).
- **`llm_judge`**: fixed `meta` cost/latency bookkeeping bug.

### Documentation
- **Doc reconciliation** (commit `54bb46d`):
  - `docs/LLM_AGENT_GUIDE.md`: test file count updated from "31+" to actual 57 / 581 test functions
  - `docs/ROADMAP.md`: qualified "509 tests" claim as Fas 4 gate snapshot, added "Known Stubs / Deferred Items" section listing `llm_judge` stub, alerting webhook TODO, and YouTube ingest rollback
  - Findings from code review session 2026-06-24 (glm-5.2 plan + grok-4.3 review)

### Security
- **PII-redaction hardening** (PR #17):
  - Luhn validation for credit cards (prevents false positives on invoice/case numbers)
  - Swedish first names list expanded from 10 → ~60 entries (SCB top)
  - Swedish address pattern expansion (suffix-based + Python validation for compound words)
  - Phone regex requires +46/0 prefix with `(?<!\d)` lookbehind (avoids matching 13-16 digit IDs)
  - CC checked before phone (priority order)
  - New `app/nicegui_dashboard/components/pii_audit.py` for PII event visibility
- **Alerting webhook production-grade** (PR #17):
  - `httpx.AsyncClient` with 10s timeout + 3 retries + exponential backoff
  - Circuit breaker: webhook disabled after 5 consecutive failures
  - Externalized config: `configs/alerting_config.yaml`
  - Env override priority: env vars > YAML > hardcoded defaults

## [0.4.1] - 2026-06-24

### Added
- **Groq Cloud LLM integration** (commit `e91edf1`, PR #16): New provider alongside Mistral/OpenRouter
  - `src/llm/groq_client.py` — `GroqClient` with OpenAI-compatible API, caching, cost tracking
  - `src/llm/groq_analyzer.py` — `GroqAnalyzer` with strict Pydantic schemas
  - `src/llm/schemas.py` — `GROQ_MODELS` registry (17 models) with pricing, capabilities, tiers
  - GDPR gate: `groq_eu_residency` config flag + pipeline enforcement + per-call anonymize check
  - CLI: `--provider groq --groq-eu-residency` (also available via API + NiceGUI dropdown)
  - Tests: `tests/test_groq_client.py` + Groq tests in `tests/test_llm_analyzer.py` (all mocked)
  - Docs: `docs/LLM_PROVIDERS.md` (full comparison matrix), updated `ROADMAP.md`, `LLM_AGENT_GUIDE.md`
  - Fallback chain: 8B → 70B → gpt-oss-20b (config-driven)
  - See `docs/LLM_PROVIDERS.md` for full pricing matrix and GDPR guidance.
- **Fas 3 – NiceGUI Dashboard (Fas 4-visualisering)**:
  - Nya flikar: **Agent Performance**, **Fas 4 Insikter**
  - Komponenter: `agent_performance.py`, `qa_scorecard.py`, `insights_hot_topics.py`, `alerts_panel.py`
  - `fas4_data.py` – lokal extraktion + API-fallback för alla 5 Fas 4-endpoints
  - `NiceGUIAPIClient` utökad med `get_agent_performance`, `semantic_search`, `get_hot_topics`, `score_qa`, `get_alerts`
  - Alerts-badge i header, drill-down till Samtalsdetalj, markera alert som hanterad (stub)
  - Tester: `tests/test_fas4_dashboard_data.py`
- **Review-fixar (Fas 3)**:
  - Async API-laddning utanför `@ui.refreshable` (undviker stale timers)
  - Korrigerad empati-aggregering i `local_agent_metrics`
  - Alerts-badge uppdaterar färg vid refresh

## [0.5-prep] - 2026-06-19 (Fas 4 Backend, never tagged as v0.4.1)

> Note: this entry originally documented work intended for v0.4.1 but
> never formally tagged. Renamed to [0.5-prep] to avoid duplicate version
> headers. The Fas 4 backend code is still present in main and will be
> released as v0.5.0 after LLM-judge + dashboard viz work lands.

### Added
- **Fas 4 – Call Center Backend (komplett)**:
  - `src/agent_performance.py` – per-call agent/customer metrics, coaching hints, aggregation
  - `src/compliance_qa.py` – YAML scorecards, hybrid rule/LLM QA scoring med evidence spans
  - `src/insights_aggregator.py` – hot topics, root cause clusters, trender
  - `src/semantic_search.py` – hybrid vektor + keyword-sökning över samtal
  - `src/alerting.py` – regelbaserade alerts med severity och recommended actions
  - `src/caching.py` – `AggregateCache` (fil/Redis), pre-computation för dashboard-frågor
- **5 nya REST API-endpoints** (Fas 4.5.2):
  - `POST /agent_performance/{agent_id}`
  - `POST /search/semantic`
  - `POST /insights/hot_topics`
  - `POST /qa/score`
  - `POST /alerts`
- **Utvärdering & validering**:
  - `python -m src.evaluate fas4-validation` – genererar `reports/evaluate_fas4_validation.md`
  - Fas 4 KPI-stubs i `evaluate.py` (QA consistency, coaching precision, hot topic recall, PII coverage, alert rate, cache hit rate)
- **Tester**: 509+ pytest-tester, 86 %+ coverage på in-scope `src/`-moduler (Fas 1 gate)
- **Dokumentation**: `docs/FAS4_COMPLETION.md`, utökad `docs/API.md`, `docs/archive/GROK_BUILD_PLAN_FAS1-3.md`

### Changed
- `CallAnalysisPipeline` mergar nu `agent_performance`, `qa`/`compliance_qa`, `pii_redaction`, `alerts` i `report.results`
- API `cached`-flagga på `/agent_performance/{agent_id}` speglar verklig cache hit (inte alltid `true`)
- `semantic_search.py`: fix för tuple/list-konkatenering i index metadata-filter

### Fixed
- Cache-test assertion (`cache_hit` metadata skiljer sig avsiktligt mellan första och andra anrop)
- Audio benchmark smoke-test mockar ML-deps-kontroll korrekt
- Keyword-boost i semantic search hanterar icke-sträng `topics`

### Documentation
- README: ny sektion "Funktioner (v0.4+)" med Fas 4 CLI/API-exempel
- ROADMAP & UTVECKLINGSPLAN: Fas 4 markerad som **Slutförd**
- Nästa steg: data/finetuning, NiceGUI dashboard-visualisering (Fas 3 i GROK-plan), produktion

## [0.4.0] - 2026-06-12

### Added
- `diarize` optional dependency group in `pyproject.toml` (`pyannote.audio>=3.1.0` + `torchaudio>=2.2.0`) to properly support Speaker Diarization feature.
- `CHANGELOG.md` for tracking releases and changes.
- Improved cross-platform Quickstart and Installation instructions in README (unified `pip install -e "[cli,api,diarize]"` recommendation + Docker example + Hardware Requirements).
- Hardware Requirements section in README (GPU/VRAM guidance for ASR and diarization).

### Changed
- Bumped version from 0.3.0 → **0.4.0** to better reflect the advanced Call Center Intelligence features (Fas 1–3, Mistral LLM integration, full `CallAnalysisPipeline`, API v0.4.0 capabilities).
- Updated project description and keywords in `pyproject.toml` to include "call-center" and "diarization".
- Clarified that `pyannote.audio` now has an official installation path (previously missing from dependency declarations).

### Fixed
- Missing dependency declaration for the Speaker Diarization feature (critical bug fix).
- Inconsistent installation experience between Windows-focused scripts and modern Python packaging (`pyproject.toml` optional deps).

### Documentation
- Added prominent Quickstart section with tiered installation (basic CLI → full call center + diarization → API).
- Added note about `ffmpeg` requirement for `--preprocess`.
- Better guidance on when to use `--use-mistral-llm` and privacy implications.

## [0.3.0] - Previous

- Initial public structure with core sentiment, ASR (faster-whisper), CLI, basic pipeline and evaluation framework.
- Introduction of `CallAnalysisPipeline`, analysis registry, profiles and Streamlit dashboard.
- Mistral / OpenRouter LLM integration (Fas 3) with structured output and caching.

See git history for earlier development details.