# AGENTS.md

This repository is optimized for LLM coding agents (Grok Build, Claude, Cursor, Aider, etc.).

**Primary entry point for agents:**

→ **[docs/LLM_AGENT_GUIDE.md](docs/LLM_AGENT_GUIDE.md)**

This guide is the single source of truth and contains:
- Project architecture and philosophy (registry pattern, graceful degradation, PII-first)
- Recommended patterns for extending the system (new Analyzer, new LLM provider, new dashboard component, new API endpoint)
- Playbooks for common tasks (adding Fas X feature, evaluation, fine-tuning)
- Coding standards, security rules, and what to do / NOT to do
- Swedish localization and call-center domain specifics

**Also read:**
- `docs/ROADMAP.md` – Current maturity and completed features
- `README.md` – Quickstart and overview
- `SECURITY.md` – Important privacy and security considerations for call center data

**Always read `docs/LLM_AGENT_GUIDE.md` before proposing or making code changes.**

## Quickstart for AI Agents

```bash
# 1. Clone and setup
pip install -e ".[cli,api,dev]"

# 2. Download ASR models (required for transcription)
sentimentanalys download-asr

# 3. Run tests to verify
pytest --tb=no -q

# 4a. Start backend API
uvicorn src.api:app --reload

# 4b. Start web UI (Next.js)
cd webui && npm install && npm run dev   # → http://localhost:3000
```

## Frontend (web UI)

The primary frontend lives in `webui/` (Next.js 16 + React 19 + TypeScript +
Tailwind v4 + shadcn/ui patterns). It talks to the existing FastAPI backend
in `src/api/` without backend changes.

- `webui/src/app/` – App Router pages
- `webui/src/lib/api/client.ts` – typed API client (`ApiClient`, `ApiError`)
- `webui/src/hooks/` – React Query hooks + WebSocket transcription client
- `webui/src/components/` – UI primitives + feature components
- `webui/e2e/` – Playwright smoke tests
- `webui/Dockerfile` + `docker-compose.webui.yml` – standalone Next.js build

## Key Files & Commands

- `pyproject.toml` — Dependencies, optional groups, scripts. Use `pip install -e ".[dev]"` for full dev setup.
- `Makefile` — Common tasks (lint, test, format).
- `src/pipeline.py` + `src/analysis/registry.py` — Core orchestration. New analysis steps go here.
- `src/llm/` — LLM clients & analyzers (add new provider here).
- `webui/` — Primary frontend (Next.js). `npm run dev` / `lint` / `build` / `test:e2e`.
- `launcher/` — Windows PowerShell launcher & ASR management.
- `tests/` — Unit + integration tests. Run with `pytest`. Full runbook: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md). Release path: [docs/PRODUCTION_CHECKLIST.md](docs/PRODUCTION_CHECKLIST.md).

## Coding Conventions (Python / FastAPI / Next.js)

- Strict Pydantic models for all I/O and LLM output (see `src/llm/schemas.py`).
- Type hints + docstrings everywhere.
- Registry pattern for extensibility (see `src/analysis/registry.py`).
- Graceful degradation for optional dependencies (ASR, diarization, LLM).
- PII/GDPR-first: redact before LLM calls when flag is set.
- Swedish-first: prompts, lexicon, data, UI strings.
- Commit style: Conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`).
- Tests: Unit + integration. Mock heavy external calls (LLM, ASR). High coverage on `src/`.

## Documentation & Context

- [CHANGELOG.md](CHANGELOG.md) & [docs/ROADMAP.md](docs/ROADMAP.md) — History and future plans.
- `docs/` — Architecture, LLM guide, API docs, production checklist.

## Security & Privacy

See [SECURITY.md](SECURITY.md). Call center data is sensitive — never commit real customer audio or PII. Use demo/fake data for development.

## When Adding Features

1. Read `docs/LLM_AGENT_GUIDE.md` and `docs/ROADMAP.md`.
2. Prefer extending the analyzer registry / API routers over one-off scripts.
3. Add tests next to the change.
4. Update `CHANGELOG.md` and `docs/ROADMAP.md` when behavior or maturity changes.
