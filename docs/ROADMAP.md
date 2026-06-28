# Roadmap & Current Status

This document provides a high-level overview of the project's maturity and future direction.

## Current Status (June 2026)

**Version**: 0.4.1 (v0.5-prep)

The project has reached a **mature beta / early production** stage. **Fas 4 (Call Center Backend) is complete** and validated (Fas 1 gate: 509 tests, 86 %+ coverage). Release documentation updated in CHANGELOG, README, API.md, and FAS4_COMPLETION.md.

> **Note:** Test count is a snapshot at Fas 4 sign-off. Current count is **581 test functions across 57 test files** (delta includes Groq integration, transcription presets, dashboard tests, and PII coverage added post-Fas 4). See `pytest --collect-only` for live count.

### Known Gaps / Deferred Items (v0.4.1)

| Component | Status | Note |
|-----------|--------|------|
| `src/analysis/llm_judge.py` | ✅ Implemented | Low-confidence routing with budget guard; enable via analyzer profile / `analyzer_configs`. |
| `src/alerting.py` webhook | ✅ Implemented | `notify_webhook()` POSTs via `httpx` with retry + circuit breaker (`configs/alerting_config.yaml`). |
| YouTube ingest (Fas 5) | ❌ Removed | Rolled back in commit `46bc04c` (experimental, not re-introduced in v0.5 scope). |
| Pipeline size | 🟡 Tech debt | `CallAnalysisPipeline` (~830 lines) — refactor into explicit steps (see `docs/CLEANUP_PLAN.md`). |
| Analyzer DX | 🟡 Tech debt | 24+ analyzers; no `new-analyzer` CLI template yet (see `docs/CLEANUP_PLAN.md` EXT-01). |

### Completed Features

| Area                        | Status     | Key Components                                      |
|-----------------------------|------------|-----------------------------------------------------|
| **Core Sentiment**          | ✅ Done    | `sentiment.py`, lexicon blending, negation handling |
| **ASR (Speech-to-Text)**    | ✅ Done    | `faster-whisper` (default), Transformers, WhisperX backends + `preprocess.py` |
| **Speaker Diarization**     | ✅ Done    | `pyannote.audio` + strong heuristic/energy-based fallback in `diarization.py` |
| **Call Analysis Pipeline**  | ✅ Done    | `CallAnalysisPipeline` with full orchestration      |
| **Analysis Registry**       | ✅ Done    | Aspect, Emotion, Role, Trajectory, Intent, Summary, Topics, Spoken Normalizer |
| **Mistral LLM Integration** | ✅ Done    | Hybrid local + Mistral via OpenRouter, structured output, caching, privacy logging |
| **Groq Cloud Integration**  | ✅ Done    | `GroqClient` + `GroqAnalyzer`, 17-model registry, GDPR gate, pricing tracking |
| **Agent Performance**       | ✅ Done    | `agent_performance.py`, cached metrics              |
| **QA & Compliance**         | ✅ Done    | `compliance_qa.py`, YAML scorecards, hybrid scoring |
| **Insights & Search**       | ✅ Done    | `insights_aggregator.py`, `semantic_search.py` (FAISS) |
| **Alerting**                | ✅ Done    | `alerting.py`, per-call and aggregate alerts        |
| **PII Protection**          | ✅ Done    | Early redaction in pipeline for `callcenter` profile |
| **REST API (v0.4.1)**       | ✅ Done    | Full FastAPI + 5 Fas 4 endpoints, auth, rate limit, caching |
| **CLI**                     | ✅ Done    | Rich `typer` CLI with `sentiment`, `transcribe`, `analyze-call` |
| **Dashboard**               | ✅ Done    | NiceGUI standard (`app/nicegui_dashboard/`); Streamlit avvecklad |
| **Evaluation Framework**    | ✅ Done    | `evaluate.py` + `fas4-validation` + LLM quality metrics |
| **Fas 4 Backend**           | ✅ Done    | Agent perf, QA, insights, search, alerts, caching (validated) |

### Partially / Recently Added

- `diarize` optional dependency group (`pyproject.toml`)
- Consolidated Quickstart + Hardware Requirements in README
- `CHANGELOG.md`, `SECURITY.md`, `CONTRIBUTING.md`, `docs/ROADMAP.md`

## Architecture Principles

- **Hybrid first**: Local models + heuristics are the fast/cheap/private path. LLM (Mistral via OpenRouter) is used selectively for high-value reasoning.
- **Graceful degradation**: Missing optional dependencies (pyannote, whisperx, etc.) fall back automatically.
- **Privacy by design**: Explicit logging of external LLM calls, PII redaction, no hardcoded secrets.
- **Extensibility**: Registry-based analyzers and clear plugin points.

## Next Priorities (post-Fas 4)

| Priority | Area                        | Description                                                                 |
|----------|-----------------------------|-----------------------------------------------------------------------------|
| High     | **Dashboard (Fas 3)**         | NiceGUI-vyer: Agent Performance, QA Scorecard, Hot Topics, Alerts, Search  |
| High     | **Data & Finetuning (Fas 2)** | Domänanpassning, utökad testkorpus, WER/sentiment-förbättring             |
| High     | Production                  | GPU Docker, observability, prod rate limiting, secrets management          |
| Medium   | Pipeline Refactoring        | Reduce complexity in `CallAnalysisPipeline` (more explicit steps)          |
| Medium   | Observability               | Structured logging, Prometheus metrics, better tracing for long calls      |
| Low      | Fine-tuning UX              | Make `finetune.py` easier to use for domain adaptation on call center data |

## Long-term Vision

Build a complete, self-hosted or hybrid **Swedish Call Center Intelligence Platform** that can compete with international solutions while keeping data in Sweden/EU and supporting the Swedish language exceptionally well.

## Historical Plans

Older detailed plans are in `docs/archive/` (see `docs/archive/README.md`). Active roadmap: **this file**.