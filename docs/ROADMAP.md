# Roadmap & Current Status

This document provides a high-level overview of the project's maturity and future direction.

## Current Status (July 2026)

**Version**: 0.5.1

The project has reached **v0.5 production-ready beta**. Fas 4 (Call Center Backend) is complete. v0.5 adds DATA-01 import workflow, Docker staging with observability, model A/B compare (API + webui), INSIGHT-02 test coverage, CI mypy/staging gates, and GPU verification tooling.

> **Note:** The test suite is continuously extended. Use `pytest --collect-only -q` for the authoritative current count; CI is the merge gate.

### Pilot direction (2026-09, decision owner: Oscar Delerud)

Aktuell utvecklingsriktning är en **operatörspilot** (Läge A) på Windows 11/RTX 5070 12 GB: QA-stöd och coachning som primärnytta, kundorganisation identifierad från ljudfilens namn, lokal CUDA-ASR som baslinje och per-kundprofil valbar lokal/extern bearbetning med lokal artefaktpersistens. Fullständigt beslutsregister: [PILOT_RUNBOOK.md](PILOT_RUNBOOK.md) §0. Filnamnsformat, faktiska kundmappningar, leverantörsval, budget och retention är öppna beslut som blockerar skarp aktivering — inte dokumentation eller syntetisk verifiering. Implementationsstatus för kundstyrd bearbetning redovisas per etapp här i takt med att den landar.

### Known Gaps / Deferred Items (v0.5.1)

| Component | Status | Note |
|-----------|--------|------|
| Real telephony corpus (DATA-01) | Open | Import slot ready; needed for customer WER/F1 claims |
| Intent model vs heuristic | Open | Promote only if model beats heuristic by +0.05 macro F1 |
| YouTube ingest (Fas 5) | Removed | Out of scope (`46bc04c`) |
| `TranscriptionEventHub` Redis | ✅ | Pub/sub when `API_USE_REDIS_CACHE=true` |
| Engine / adapter split | Documented | Logic in `src/*.py`, registry adapters in `src/analysis/` |

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
| **REST API (v0.5.1)**       | ✅ Done    | Full FastAPI + Fas 4 endpoints, auth, rate limit, caching |
| **CLI**                     | ✅ Done    | Rich `typer` CLI with `sentiment`, `transcribe`, `analyze-call` |
| **Dashboard**               | ✅ Done    | Next.js webui (`webui/`) — sole dashboard |
| **Evaluation Framework**    | ✅ Done    | `evaluate.py` + `fas4-validation` + LLM quality metrics |
| **HTTP Metrics (OBS-01)**     | ✅ Done    | `http_requests_total`, `http_request_duration_seconds` in `src/api/metrics.py` |
| **Pipeline Refactoring**      | ✅ Done    | PIPE-01: `pipeline_steps.py`; orchestration in `pipeline.py` (~725 LOC) |

### Partially / Recently Added

- **Våg 2 (2026-09-12)** — QA delar client-resolver med holistic; LM Studio-UI bara vid DIRECT_API; Testlabb visar serversegment efter PII.
- **LM Studio lokal LLM-integration (2026-09)** — `lmstudio` provider för Qwen 3.5 9B The Defiant (Q4_K_S) vid 70k total kontext; delad klientfactory för holistisk/QA/judge/insights; strukturerad utdata + evidensvalidering; beständiga bakgrundsjobb (`/analysis/jobs`); WebUI-panel + CLI `--provider lmstudio`; CPU-ASR-alternativ. Reasoning-off kan inte garanteras via OpenAI-kompatibelt API (se `docs/LM_STUDIO_LOCAL.md`). Ej pilot-godkänd.
- **ASR dual-engine hardening (2026-07)** — router, cloud opt-in, decode hardening, persistent jobs, metrics; see `CHANGELOG.md`.
- `diarize` optional dependency group (`pyproject.toml`)
- Consolidated Quickstart + Hardware Requirements in README
- `CHANGELOG.md`, `SECURITY.md`, `CONTRIBUTING.md`, `docs/ROADMAP.md`

## Architecture Principles

- **Hybrid first**: Local models + heuristics are the fast/cheap/private path. LLM (Mistral via OpenRouter) is used selectively for high-value reasoning.
- **Graceful degradation**: Missing optional dependencies (pyannote, whisperx, etc.) fall back automatically.
- **Privacy by design**: Explicit logging of external LLM calls, PII redaction, no hardcoded secrets.
- **Extensibility**: Registry-based analyzers and clear plugin points.

## Strategy & decision pack (2026-07, updated 2026-09)

Canonical product strategy: **[STRATEGY.md](../STRATEGY.md)**.  
Executive go/no-go + 90-day plan: **[docs/DECISION_REPORT_2026-07-17.md](DECISION_REPORT_2026-07-17.md)** (historiskt).  
Operational pilot locks + beslutsregister R01–R09: **[docs/PILOT_RUNBOOK.md](PILOT_RUNBOOK.md)** · corpus spec: **[docs/DATA_01_CORPUS_SPEC.md](DATA_01_CORPUS_SPEC.md)**.  
Frontend ↔ backend harmony: **[docs/FE_BE_HARMONY_2026-07-17.md](FE_BE_HARMONY_2026-07-17.md)**.

**Verdict (2026-07, Läge B):** *conditional go* for a controlled pilot (local ASR, anonymize LLM, Groq off, DATA-01 + L7–L9).  
**Riktning (2026-09, Läge A):** operatörspilot enligt [PILOT_RUNBOOK.md](PILOT_RUNBOOK.md) §0.

## Next Priorities (post v0.5)

| Priority | Area | Description |
|----------|------|-------------|
| High | **Operatörspilot — kundkontext** (Läge A) | Kund-ID-resolver från originalfilnamn, versionssatt kundregister i install-schema, fryst kundkonfiguration per jobb (Etapp 2 i planen) |
| High | **Operatörspilot — lokal datalivscykel** (Läge A) | Beständiga artefakter/transkript/rapporter per kund, bakgrundsjobbet äger kundflödet, idempotens per kund+källa+config (Etapp 3) |
| High | **Operatörspilot — exekveringspolicy** (Läge A) | Kundstyrd ASR/LLM-routing med strikt tillåtelselista, klassificerad fallback, gemensam försök-/budgetkontext (Etapp 4) |
| High | **Operatörspilot — GPU/resurser** (Läge A) | Gemensam resursreservation ASR/LLM, verifierad CUDA-enhet och modellavlastning på RTX 5070 (Etapp 5) |
| High | **Real corpus** | Replace synthetic DATA-01 bundle (`generate_pilot_corpus.py`) with anonymized telephony via `--pilot-gate` — [DATA_01_CORPUS_SPEC.md](DATA_01_CORPUS_SPEC.md) |
| High | **Pilot release gates** | Run `scripts/run_pilot_gates.py` (L7 fixture + policy); close L8/L9 + live `/testlab` before customer pilot — [PILOT_RUNBOOK.md](PILOT_RUNBOOK.md) |
| High | **Intent fine-tune** | Promote model that beats heuristic + 0.05 macro F1 (`train_intent_smoke.py` / `train_intent.py` + `run_quality_gates.py`) |
| Medium | **OTLP tracing** | Replace console OTEL exporter with production OTLP endpoint |
| Medium | **Dashboard polish** | Correlation heatmap, executive drill-downs; kundkontext i UI (Etapp 6) |
| Low | **Fine-tuning UX** | Easier domain adaptation workflow for call center data |

### Ops hardening landed (post-genomlysning)

- Server-side `/calls` persistence + webui sync (localStorage = cache)
- Next.js BFF proxy (`NEXT_PUBLIC_USE_API_PROXY`) to hide API keys
- `/ready` readiness + `/status/health/detail` degraded checks
- Pipeline `degraded` / `mode` fields + UI banner
- Redis-backed WS hub visibility + staging auth/media-root locked
- Pilot corpus generator + quality-gate orchestrator

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

Active roadmap: **this file**. Historical plan docs were removed; see `CHANGELOG.md` for history.