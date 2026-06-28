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

```bash
make test                 # All tests
make test-api             # API tests with coverage
python -m src.evaluate llm-quality
```

### Audio benchmarks (`samples/audio`)

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
- `OTEL_ENABLED=true` – Optional OpenTelemetry tracing

## Fine-tuning (DATA-01)

```bash
pip install -e ".[training,min]"
python scripts/prepare_callcenter_data.py --target-size 10000
python scripts/validate_domain_corpus.py data/callcenter_val.csv
python -m src.finetune --config configs/finetune.yaml
```

CI runs smoke tests via `configs/finetune.ci.yaml`. Baseline: `reports/finetune_baseline.json`.

When `models/callcenter-sentiment-lora/` exists, `callcenter` profile uses it automatically.

## Importera riktig domändata (GDPR)

Never commit real customer audio or transcripts. See [SECURITY.md](../SECURITY.md).

1. Store raw data outside the repo (encrypted volume / secure bucket)
2. Anonymize PII before labelling (use pipeline PII redaction as reference)
3. Export CSV with columns `text,label` (negativ/neutral/positiv)
4. Validate with `python scripts/validate_domain_corpus.py your_val.csv`
5. Run `python -m src.evaluate --testset your_val.csv --output reports/domain_eval.json`

Synthetic data from `scripts/prepare_callcenter_data.py` is for development only.

## Before Committing / Creating a PR

1. Run `make check`
2. Run `make test`
3. Update relevant documentation (`README.md`, `docs/ROADMAP.md`, `CHANGELOG.md`)
4. If adding a new analyzer or major feature, consider updating `docs/LLM_AGENT_GUIDE.md`

## Related Documentation

- `AGENTS.md` – Entry point for LLM coding agents
- `docs/LLM_AGENT_GUIDE.md` – Detailed guide for agents
- `docs/LLM_AGENT_QUICKREF.md` – Minimal context quick reference
- `docs/ROADMAP.md` – Current project status
- `SECURITY.md` – Security and privacy guidelines