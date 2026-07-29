# Roadmap & Current Status

This document provides a high-level overview of the project's maturity and future direction.

## Current Status (July 2026)

**Version**: 0.5.0

The project has reached **v0.5 production-ready beta**. Fas 4 (Call Center Backend) is complete. v0.5 adds DATA-01 import workflow, Docker staging with observability, model A/B compare (API + webui), INSIGHT-02 test coverage, CI mypy/staging gates, and GPU verification tooling.

> **Note:** The test suite is continuously extended. Use `pytest --collect-only -q` for the authoritative current count; CI is the merge gate.

### Known Gaps / Deferred Items (v0.4.1)

| Component | Status | Note |
|-----------|--------|------|
| `src/analysis/llm_judge.py` | ✅ Implemented | Low-confidence routing with budget guard; enable via analyzer profile / `analyzer_configs`. |
| `src/alerting.py` webhook | ✅ Implemented | `notify_webhook()` POSTs via `httpx` with retry + circuit breaker (`configs/alerting_config.yaml`). |
| YouTube ingest (Fas 5) | ❌ Removed | Rolled back in commit `46bc04c` (experimental, not re-introduced in v0.5 scope). |
| Pipeline size | ✅ Refactored | Fas-4/LLM in `pipeline_steps.py` (PIPE-01); `pipeline.py` < 550 LOC |
| Analyzer DX | ✅ | `sentimentanalys new-analyzer` CLI template |
| Dependencies | ✅ | `pyproject.toml` only (DEPS-01); no `requirements*.txt` |

### Completed Features

| Area                        | Status     | Key Components                                      |
|-----------------------------|------------|-----------------------------------------------------|
| **Core Sentiment**          | ✅ Done    | `sentiment.py`, lexicon blending, negation handling |
| **ASR (Speech-to-Text)**    | ✅ Done    | `AsrRouter` (local default), `faster-whisper` / Transformers / WhisperX, opt-in Deepgram, hallucination filter, chunk retry, persistent jobs + metrics |
| **Speaker Diarization**     | ✅ Done    | `pyannote.audio` when HF token present + heuristic/energy fallback in `diarization.py` |
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
| **Dashboard**               | ✅ Done    | Next.js webui (`webui/`) — sole dashboard |
| **Evaluation Framework**    | ✅ Done    | `evaluate.py` + `fas4-validation` + LLM quality metrics |
| **HTTP Metrics (OBS-01)**     | ✅ Done    | `http_requests_total`, `http_request_duration_seconds` in `src/api/metrics.py` |
| **Pipeline Refactoring**      | ✅ Done    | PIPE-01: `pipeline_steps.py`; `pipeline.py` < 550 LOC |

### Partially / Recently Added

- **ASR dual-engine hardening (2026-07)** — router, cloud opt-in, decode hardening, persistent jobs, metrics; see `CHANGELOG.md` Unreleased.
- `diarize` optional dependency group (`pyproject.toml`)
- Consolidated Quickstart + Hardware Requirements in README
- `CHANGELOG.md`, `SECURITY.md`, `CONTRIBUTING.md`, `docs/ROADMAP.md`

## Architecture Principles

- **Hybrid first**: Local models + heuristics are the fast/cheap/private path. LLM (Mistral via OpenRouter) is used selectively for high-value reasoning.
- **Graceful degradation**: Missing optional dependencies (pyannote, whisperx, etc.) fall back automatically.
- **Privacy by design**: Explicit logging of external LLM calls, PII redaction, no hardcoded secrets.
- **Extensibility**: Registry-based analyzers and clear plugin points.

## Strategy & decision pack (2026-07)

Canonical product strategy: **[STRATEGY.md](../STRATEGY.md)**.  
Executive go/no-go + 90-day plan: **[docs/DECISION_REPORT_2026-07-17.md](DECISION_REPORT_2026-07-17.md)**.  
Operational pilot locks: **[docs/PILOT_RUNBOOK.md](PILOT_RUNBOOK.md)** · corpus spec: **[docs/DATA_01_CORPUS_SPEC.md](DATA_01_CORPUS_SPEC.md)**.  
Frontend ↔ backend harmony: **[docs/FE_BE_HARMONY_2026-07-17.md](FE_BE_HARMONY_2026-07-17.md)**.

**Verdict:** *conditional go* for a controlled pilot (local ASR, anonymize LLM, Groq off, DATA-01 + L7–L9).

## Next Priorities (post v0.5)

| Priority | Area | Description |
|----------|------|-------------|
| High | **Real corpus** | Import anonymized production calls via DATA-01 (`--pilot-gate`: ≥500 sentiment / ≥200 intent) — [DATA_01_CORPUS_SPEC.md](DATA_01_CORPUS_SPEC.md) |
| High | **Pilot release gates** | Close L7–L9 + `verify_pilot_policy.py` before customer pilot — [PILOT_RUNBOOK.md](PILOT_RUNBOOK.md) |
| High | **Intent fine-tune** | Train model that beats heuristic + 0.05 macro F1 on `intent_val.jsonl` (training/config/auto-backend plumbing ready; local artifact and measured promotion gate remain) |
| Medium | **OTLP tracing** | Replace console OTEL exporter with production OTLP endpoint |
| Medium | **Dashboard polish** | Correlation heatmap, executive drill-downs |
| Low | **Fine-tuning UX** | Easier domain adaptation workflow for call center data |

### Completed in v0.5

- DATA-01 import workflow + baseline eval smoke + CI gates
- Docker staging (`docker-compose.staging.yml`) with Redis cache + Prometheus
- Model A/B compare: `POST /analyze_pipeline/compare` + Testlabb panel
- INSIGHT-02 deep-path skip tests
- CI: mypy job + staging compose config validation
- GPU Docker verification script + checklist documentation

## Long-term Vision

Build a complete, self-hosted or hybrid **Swedish Call Center Intelligence Platform** that can compete with international solutions while keeping data in Sweden/EU and supporting the Swedish language exceptionally well.

## Historical Plans

Older detailed plans are in `docs/archive/` (see `docs/archive/README.md`). Active roadmap: **this file**.