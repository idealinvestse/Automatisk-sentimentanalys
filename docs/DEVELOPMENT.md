# Development Guide

This document describes how to develop and contribute to **Automatisk-sentimentanalys**.

## Quick Start for Developers

```bash
# 1. Clone and setup
make install-dev          # or: pip install -e ".[dev,diarize]"

# 2. Run quality checks
make check

# 3. Run tests
make test
```

## Available Make Targets

Run `make help` to see all available commands.

Most useful targets:

| Command            | Description                              |
|--------------------|------------------------------------------|
| `make install-dev` | Install with dev + diarize extras        |
| `make test`        | Run all tests                            |
| `make check`       | Lint + format check + mypy               |
| `make format`      | Auto-format code                         |
| `make lint`        | Run ruff linting                         |
| `make run-api`     | Start FastAPI development server         |
| `make run-dashboard` | Start Streamlit dashboard              |
| `make clean`       | Remove cache and build artifacts         |

## Code Quality

We enforce code quality using:

- **Ruff** (linting + formatting)
- **Mypy** (type checking)
- **Pre-commit** hooks

### Using Pre-commit

```bash
# Install pre-commit hooks (once)
pre-commit install

# Run manually on all files
make pre-commit
# or
pre-commit run --all-files
```

## Project Structure Highlights

See `docs/LLM_AGENT_GUIDE.md` for a detailed breakdown aimed at both humans and LLM agents.

Key principles:
- **Registry pattern** for analyzers (`src/analysis/registry.py`)
- **Graceful degradation** for optional heavy dependencies
- **Hybrid architecture** (local first, LLM when needed)

## Running the Application

### CLI
```bash
python -m src.cli --help
python -m src.cli analyze-call samples/audio/sv/ --backend faster --language sv
```

### API
```bash
make run-api
# or
uvicorn src.api:app --reload
```

### Dashboard
```bash
make run-dashboard
```

## Testing

Full-stack test strategy for this repo. Goal: maximum confidence per minute spent.
CI is the floor (fast, deterministic); release/staging covers what CI deliberately skips
(real ASR, live LLM, live dashboard ↔ API).

### Principles

1. **Risk over coverage** — prefer product-breaking paths (wrong sentiment/intent, PII to LLM, broken pipeline/UI) over chasing % on trivial code.
2. **Mock expensive, measure quality separately** — ML weights and external LLMs are mocked in the fast suite; real quality is gated by accuracy/eval scripts.
3. **One primary proof per layer** — unit, golden pipeline, API contract, corpus F1, or UI smoke.
4. **Extend existing patterns** — golden fixtures, `configs/analyzer_eval.yaml`, Playwright with stubbed backend, jobs in `.github/workflows/ci.yml`.

### Layers (L0–L9)

| Layer | Purpose | Primary proof | When |
|-------|---------|---------------|------|
| L0 Lint/types | Static quality | ruff, black, mypy | before commit/PR |
| L1 Unit | Isolated logic | `tests/test_*.py` (no ML load) | every change in that module |
| L2 Registry/pipeline | Orchestration + deps | `test_analysis_registry`, `test_pipeline`, `test_insight02_deep_path` | analyzer/pipeline changes |
| L3 Golden | Call-center domain regression | `test_callcenter_golden` + `tests/fixtures/callcenter_golden/` | profile/heuristic/pipeline changes |
| L4 API contract | HTTP, auth, error shape | `test_api*`, `tests/contracts/` | API/schema changes |
| L5 Quality gates | Intent/analyzer F1, sentiment gate | `scripts/benchmark_*`, `eval_sentiment_gate` | corpus/heuristic/model changes |
| L6 Webui | Routes + critical UI | Playwright `webui/e2e/` | webui changes |
| L7 Heavy integration | Real ASR/audio | `@pytest.mark.audio` / `src.evaluate audio` | before release |
| L8 LLM quality | Live provider (cost) | `python -m src.evaluate llm-quality` | before release with LLM on |
| L9 Staging | Compose + observability | `docker-compose.staging.yml` + smoke script | deploy candidate |

Do **not** put L7/L8 into the PR CI floor — keep them on the release path (see [PRODUCTION_CHECKLIST.md](PRODUCTION_CHECKLIST.md) § Release verification).

### When to run what

**A. Daily development (minutes)**

```bash
make check
pytest -m "not slow" -q
# plus targeted files for what you touched, e.g.:
pytest tests/test_pipeline.py tests/test_callcenter_golden.py -q
```

| You changed… | Also run |
|--------------|----------|
| `@register_analyzer` | registry + relevant quality/heuristic test + one golden scenario |
| API | `make test-api` (prefer CI’s 90% gate on `src/api`) |
| Webui | `cd webui && npm run lint && npm run test:e2e` |

**B. Pull request (CI floor)** — already in `.github/workflows/ci.yml`:

1. lint → pytest (3.11/3.12, `src` cov ≥80 %) → api-test (`src/api` cov ≥90 %)
2. mypy, docker config/build
3. analyzer-accuracy + finetune-smoke
4. webui lint/build/e2e

**C. Before merging ML / heuristic PRs (beyond CI)**

```bash
pytest tests/test_analyzer_quality.py tests/test_callcenter_golden.py -q
# on corpus changes: same validate + benchmark as CI analyzer-accuracy
python -m src.evaluate llm-quality   # only if deep-path/LLM routing changed and a key is set
```

Never lower quality thresholds without documenting why in the PR; prefer growing the corpus first.

**D. Release candidate (full-stack confidence)** — checklist in [PRODUCTION_CHECKLIST.md](PRODUCTION_CHECKLIST.md) § Release verification (L7–L9).

### Minimum before merge (by change type)

| Change type | Minimum |
|-------------|---------|
| Pure utility | targeted unit tests + fast suite |
| New/changed analyzer | registry + quality/heuristic or labeled fixture; golden if call-center impact |
| Pipeline / deep-path | `test_pipeline` + `test_insight02_deep_path` + golden |
| API | API tests with 90% gate (local or CI) |
| Corpus / thresholds | validate + benchmark gates |
| Webui | lint + build + Playwright |
| Release | L7–L9 |

### Quick commands

```bash
make test                 # All tests
make test-api             # API tests with coverage
pytest -m "not slow" -q   # Fast local loop
python -m src.evaluate llm-quality
```

**Golden mock discipline:** when patching sentiment in pipeline/golden tests, patch
`SentimentAnalyzer._get_pipeline` (not only `.analyze`) so silent fallback cannot hide regressions.

### Audio benchmarks (`samples/audio`) — L7

The repo ships RAVDESS English speech files under `samples/audio/Actor_*` (1440 `.wav` files).
A manifest-driven catalog in `samples/audio/manifest.yaml` powers structured ASR and pipeline tests.

```bash
# Catalog overview and validation (fast, no ML)
python -m src.evaluate audio list --pack ravdess_en --limit 10
python -m src.evaluate audio validate

# Quick smoke (3 curated files; use --dry-run to preview selection only)
python -m src.evaluate audio smoke --device cpu
python -m src.evaluate audio run --scenario pipeline --pack ravdess_en --limit 2 --device cpu
```

**Adding Swedish test files:** place audio under `samples/audio/sv/<category>/` and optional
`filename.meta.yaml` sidecars. See `samples/audio/sv/README.md`, then enable the pack in
`manifest.yaml` and run `python -m src.evaluate audio validate`.

**Pytest:** fast catalog tests always run; slow ASR integration tests are marked `audio` + `slow`.
Skip them with `SENTIMENT_SKIP_AUDIO=1` or `pytest -m "not slow"`.

CPU smoke on 3 files typically takes several minutes on first run (model download + ASR).
GPU significantly speeds up ASR and pipeline scenarios.

### Known gaps (backlog, not PR blockers)

Ordered by risk × cost:

1. Real anonymized call corpus (DATA-01) — import slot ready, awaiting external data
2. Intent fine-tune model under `models/intent_classifier` (must beat heuristic +0.05 F1)
3. Swedish ASR audio pack under `samples/audio/sv/` (dirs exist, no wav yet)
4. Webui Vitest/RTL for hooks/API client — keep Playwright thin
5. Staging live-API smoke as mandatory pre-deploy step
6. WS event hub Redis pub/sub (tickets already Redis-capable via `TicketStore`)
7. Load/concurrency — defer until multi-worker pressure

Out of optimal strategy for now: chasing coverage on the omit list (CLI, whisper backends, `secrets_win`); expanding archived NiceGUI tests.

## Adding New Features

Please follow the guidelines in:
- `docs/LLM_AGENT_GUIDE.md` (especially "How to Extend the System")
- `CONTRIBUTING.md`

## Environment Variables

Common variables:

- `OPENROUTER_API_KEY` – Required for Mistral LLM features
- `HF_TOKEN` – Required for pyannote diarization models
- `SENTIMENT_API_KEY` – Enables API authentication
- `API_PRODUCTION` / `API_REQUIRE_AUTH` / `API_REQUIRE_MEDIA_ROOT` – Production guards (v0.5)
- `SENTIMENT_JSON_LOGS=1` – Structured JSON logging
- `LOG_LEVEL` / `SENTIMENT_LOG_LEVEL` – Log verbosity (`DEBUG` in dev via `SENTIMENT_DEV=1`, `INFO` in prod)
- `SENTIMENT_STATUS_FILE=1` – Append process status events to `.cache/process_events.jsonl`
- `SENTIMENT_STATUS_RING_SIZE` – In-memory status ring buffer size (default 1000)
- `SENTIMENT_STATUS_FILE_MAX_BYTES` – Rotating JSONL max size (default 5MB)
- `SENTIMENT_STATUS_DEDUP_WINDOW_S` – Dedup identical status events (0=off, e.g. 5 for prod)
- `SENTIMENT_LOG_SAMPLE` – DEBUG sampling, e.g. `registry=10,asr=5` (keep every Nth)
- `SENTIMENT_LOG_COMPONENTS` – Per-logger levels, e.g. `src.transcription=DEBUG,src.llm=WARNING`
- `OTEL_ENABLED=true` – Optional OpenTelemetry tracing

## Felsökning / observability

1. **Öka verbositet lokalt**: `SENTIMENT_DEV=1` eller `--verbose` / `--log-level DEBUG` på CLI.
2. **Strukturerade loggar**: `SENTIMENT_JSON_LOGS=1` (API och alla entry points som anropar `configure_logging()`).
3. **Live status**: läs `.cache/process_events.jsonl` eller `GET /status/processes` (senaste N events).
4. **Detaljerad health**: `GET /status/health/detail` (registrerade analyzers, ASR-backends, cache).
5. **Följ ett pipeline-jobb**: filtrera JSONL/API-events på `component=pipeline` och `job_id` (sätts via `log_context(job_id=...)`).

Exempel:

```bash
# API med DEBUG
SENTIMENT_DEV=1 uvicorn src.api:app --reload

# Senaste process-events
curl -s http://localhost:8000/status/processes?limit=20 | jq .

# Filtrera på job_id eller komponent
curl -s "http://localhost:8000/status/processes?job_id=abc&component=pipeline" | jq .

# Live job-status
curl -s http://localhost:8000/status/jobs/abc | jq .

# CLI med verbose
sentimentanalys --verbose analyze-call samples/audio/sv/ --backend faster
```

## Fine-tuning (DATA-01)

```bash
pip install -e ".[training,min]"
python scripts/prepare_callcenter_data.py --target-size 10000
python scripts/prepare_intent_data.py --per-intent 35
python scripts/validate_domain_corpus.py data/callcenter_val.csv
python scripts/validate_intent_corpus.py data/intent_train.jsonl --min-rows 200
python -m src.finetune --config configs/finetune.yaml
```

**Analyzer accuracy benchmarks:**

```bash
python scripts/benchmark_intent.py --val-file data/intent_val.jsonl --min-macro-f1 0.75
python scripts/benchmark_analyzers.py --check-thresholds
python scripts/compare_intent_backends.py
python scripts/evaluate_real_corpus.py --sentiment-csv /path/to/anonymized.csv
```

CI runs smoke tests via `configs/finetune.ci.yaml`. Baselines: `reports/finetune_baseline.json`, `reports/intent_baseline.json`, `reports/analyzer_baseline.json`.

When `models/callcenter-sentiment-lora/` exists, `callcenter` profile uses it automatically.

## Importera riktig domändata (GDPR) — DATA-01 runbook

Never commit real customer audio or transcripts. See [SECURITY.md](../SECURITY.md).

### Workflow

1. **Lagring utanför repo** — rådata på krypterad volym eller säker bucket (ej i git)
2. **PII-redaktion** — kör pipeline-redaktion som referens (`callcenter`-profil, [`src/llm/pii_redactor.py`](../src/llm/pii_redactor.py)); manuell review obligatorisk
3. **Exportformat**
   - Sentiment: CSV med kolumner `text,label` (`negativ` / `neutral` / `positiv`)
   - Intent: JSONL med `{"text": "...", "intent": "..."}` per rad
4. **Import till repo-slot** (gitignored):

```bash
python scripts/import_domain_corpus.py --source-dir /secure/path/to/anonymized
# → data/import/callcenter_val_real.csv
# → data/import/intent_val_real.jsonl
```

5. **Validering och baseline**

```bash
python scripts/validate_domain_corpus.py data/import/callcenter_val_real.csv --min-rows 50
python scripts/validate_intent_corpus.py data/import/intent_val_real.jsonl --min-rows 20
python scripts/evaluate_real_corpus.py \
  --sentiment-csv data/import/callcenter_val_real.csv \
  --intent-jsonl data/import/intent_val_real.jsonl
```

6. **Uppdatera baselines** — jämför mot `reports/domain_baseline.json` och `reports/intent_baseline.json`

### Smoke utan riktig data

```bash
python scripts/validate_domain_corpus.py data/callcenter_val.csv --min-rows 500
python scripts/evaluate_real_corpus.py --sentiment-csv data/callcenter_val.csv
```

Synthetic data from `scripts/prepare_callcenter_data.py` is for development only.

## Before Committing / Creating a PR

1. Run `make check`
2. Run the minimum suite for your change type (see Testing → Minimum before merge); at least `pytest -m "not slow" -q`
3. Update relevant documentation (`README.md`, `docs/ROADMAP.md`, `CHANGELOG.md`)
4. If adding a new analyzer or major feature, consider updating `docs/LLM_AGENT_GUIDE.md`

## Related Documentation

- `AGENTS.md` – Entry point for LLM coding agents
- `docs/LLM_AGENT_GUIDE.md` – Detailed guide for agents
- `docs/LLM_AGENT_QUICKREF.md` – Minimal context quick reference
- `docs/ROADMAP.md` – Current project status
- `docs/PRODUCTION_CHECKLIST.md` – Production + release verification (L7–L9)
- `SECURITY.md` – Security and privacy guidelines