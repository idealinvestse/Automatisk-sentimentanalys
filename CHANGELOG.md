# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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