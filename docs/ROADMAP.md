# Roadmap & Current Status

This document provides a high-level overview of the project's maturity and future direction.

## Current Status (June 2026)

**Version**: 0.4.0

The project has reached a **mature beta / early production** stage. Most planned features from Fas 1–4 are implemented and integrated.

### Completed Features

| Area                        | Status     | Key Components                                      |
|-----------------------------|------------|-----------------------------------------------------|
| **Core Sentiment**          | ✅ Done    | `sentiment.py`, lexicon blending, negation handling |
| **ASR (Speech-to-Text)**    | ✅ Done    | `faster-whisper` (default), Transformers, WhisperX backends + `preprocess.py` |
| **Speaker Diarization**     | ✅ Done    | `pyannote.audio` + strong heuristic/energy-based fallback in `diarization.py` |
| **Call Analysis Pipeline**  | ✅ Done    | `CallAnalysisPipeline` with full orchestration      |
| **Analysis Registry**       | ✅ Done    | Aspect, Emotion, Role, Trajectory, Intent, Summary, Topics, Spoken Normalizer |
| **Mistral LLM Integration** | ✅ Done    | Hybrid local + `mistralai/mistral-medium-3.5`, structured output, caching, privacy logging |
| **Agent Performance**       | ✅ Done    | `agent_performance.py`, cached metrics              |
| **QA & Compliance**         | ✅ Done    | `compliance_qa.py`, YAML scorecards, hybrid scoring |
| **Insights & Search**       | ✅ Done    | `insights_aggregator.py`, `semantic_search.py` (FAISS) |
| **Alerting**                | ✅ Done    | `alerting.py`, per-call and aggregate alerts        |
| **PII Protection**          | ✅ Done    | Early redaction in pipeline for `callcenter` profile |
| **REST API (v0.4.0)**       | ✅ Done    | Full FastAPI with auth (`X-API-Key`), batch, scan, pipeline endpoints |
| **CLI**                     | ✅ Done    | Rich `typer` CLI with `sentiment`, `transcribe`, `analyze-call` |
| **Dashboard**               | ✅ Done    | Streamlit dashboard                                 |
| **Evaluation Framework**    | ✅ Done    | `evaluate.py` + LLM quality metrics                 |

### Partially / Recently Added

- `diarize` optional dependency group (`pyproject.toml`)
- Consolidated Quickstart + Hardware Requirements in README
- `CHANGELOG.md`, `SECURITY.md`, `CONTRIBUTING.md`, `docs/ROADMAP.md`

## Architecture Principles

- **Hybrid first**: Local models + heuristics are the fast/cheap/private path. LLM (Mistral via OpenRouter) is used selectively for high-value reasoning.
- **Graceful degradation**: Missing optional dependencies (pyannote, whisperx, etc.) fall back automatically.
- **Privacy by design**: Explicit logging of external LLM calls, PII redaction, no hardcoded secrets.
- **Extensibility**: Registry-based analyzers and clear plugin points.

## Next Priorities (Suggested)

| Priority | Area                        | Description                                                                 |
|----------|-----------------------------|-----------------------------------------------------------------------------|
| High     | Production Hardening        | Improve error messages, add request tracing, better rate limiting in API   |
| High     | Documentation               | Expand examples in `docs/API.md`, decision guide for LLM usage             |
| High     | GPU / Docker                | Official GPU-enabled Dockerfile + better CUDA documentation                |
| Medium   | Dependency Cleanup          | Make `pyproject.toml` the single source of truth for dependencies          |
| Medium   | Pipeline Refactoring        | Reduce complexity in `CallAnalysisPipeline` (more explicit steps)          |
| Medium   | Observability               | Structured logging, Prometheus metrics, better tracing for long calls      |
| Low      | Fine-tuning UX              | Make `finetune.py` easier to use for domain adaptation on call center data |
| Low      | Dashboard v2                | More interactive visualizations and agent comparison views                 |

## Long-term Vision

Build a complete, self-hosted or hybrid **Swedish Call Center Intelligence Platform** that can compete with international solutions while keeping data in Sweden/EU and supporting the Swedish language exceptionally well.

## Historical Plans

Older detailed plans can be found in:
- `UTVECKLINGSPLAN.md` (largely completed)
- `UTVECKLINGSPLAN_Fas4_Backend_CallCenter_Features_v1.1.md`
- `docs/PHASE*.md` and `docs/FAS*.md` files

These are kept for reference. The current `ROADMAP.md` reflects the actual implemented state.