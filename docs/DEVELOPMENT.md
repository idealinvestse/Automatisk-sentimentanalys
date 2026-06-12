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
python -m src.cli analyze-call samples/call.wav --backend faster --language sv
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

## Adding New Features

Please follow the guidelines in:
- `docs/LLM_AGENT_GUIDE.md` (especially "How to Extend the System")
- `CONTRIBUTING.md`

## Environment Variables

Common variables:

- `OPENROUTER_API_KEY` – Required for Mistral LLM features
- `HF_TOKEN` – Required for pyannote diarization models
- `SENTIMENT_API_KEY` – Enables API authentication

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