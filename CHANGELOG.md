# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Groq Cloud LLM integration**: New provider alongside Mistral/OpenRouter
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

## [0.4.1] - 2026-06-19 (v0.5-prep)

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
- **Dokumentation**: `docs/FAS4_COMPLETION.md`, utökad `docs/API.md`, `docs/GROK_BUILD_PLAN_FAS1-3.md`

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