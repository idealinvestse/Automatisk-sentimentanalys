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
- `docs/ROADMAP.md` – Current maturity and completed features (Fas 1–4)
- `docs/CLEANUP_PLAN.md` – Documentation debt and consolidation plan
- `docs/WEBUI_MODERNIZATION_PLAN.md` – Plan + progress for the new Next.js dashboard (`webui/`) replacing the NiceGUI UI
- `README.md` – Quickstart and overview
- `SECURITY.md` – Important privacy and security considerations for call center data

**Always read `docs/LLM_AGENT_GUIDE.md` before proposing or making code changes.**

## Quickstart for Grok Build / AI Agents

```bash
# 1. Clone and setup
pip install -e ".[cli,api,dashboard-nicegui]"

# 2. Download ASR models (required for transcription)
sentimentanalys download-asr

# 3. Run tests to verify
pytest --tb=no -q

# 4a. Start backend API
uvicorn src.api:app --reload

# 4b. Start web UI (primary dashboard – Next.js)
cd webui && npm install && npm run dev   # → http://localhost:3000

# 4c. Legacy NiceGUI dashboard (still available)
python -m app.archive.nicegui_dashboard.main
```

## Frontend (web UI)

The primary frontend lives in `webui/` (Next.js 16 + React 19 + TypeScript +
Tailwind v4 + shadcn/ui patterns). It talks to the existing FastAPI backend
in `src/api/` without backend changes.

- `webui/src/app/` – App Router pages (Översikt, Analys & Trender,
  Agentprestanda, Fas 4 Insikter, Transkribering, Testlabb, Call Detail).
- `webui/src/lib/api/client.ts` – typed API client (`ApiClient`, `ApiError`).
- `webui/src/hooks/` – React Query hooks + WebSocket transcription client.
- `webui/src/components/` – UI primitives (shadcn-style) + feature components.
- `webui/e2e/` – Playwright smoke tests (one per route).
- `webui/Dockerfile` + `docker-compose.webui.yml` – standalone Next.js build.

See [docs/WEBUI_MODERNIZATION_PLAN.md](docs/WEBUI_MODERNIZATION_PLAN.md) for
the full migration status. `app/archive/nicegui_dashboard/` is the archived legacy.

## Key Files & Commands

- `pyproject.toml` — Dependencies, optional groups, scripts. Use `pip install -e ".[dev]"` for full dev setup.
- `Makefile` — Common tasks (lint, test, format).
- `src/pipeline.py` + `src/analysis/registry.py` — Core orchestration. New analysis steps go here.
- `src/llm/` — LLM clients & analyzers (add new provider here).
- `app/archive/nicegui_dashboard/` — Archived legacy NiceGUI dashboard (deprecated; use `webui/`).
- `webui/` — Primary frontend (Next.js 16 + React 19 + TS + Tailwind v4). `npm run dev` / `lint` / `build` / `test:e2e`. See `docs/WEBUI_MODERNIZATION_PLAN.md`.
- `launcher/` — Windows PowerShell launcher & ASR management.
- `tests/` — 500+ tests. Run with `pytest`.

## Coding Conventions (Python / FastAPI / NiceGUI)

- Strict Pydantic models for all I/O and LLM output (see `src/llm/schemas.py`).
- Type hints + docstrings everywhere.
- Registry pattern for extensibility (see `src/analysis/registry.py`).
- Graceful degradation for optional dependencies (ASR, diarization, LLM).
- PII/GDPR-first: redact before LLM calls when flag is set.
- Swedish-first: prompts, lexicon, data, UI strings.
- Commit style: Conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`). Reference TASK- or Fas- numbers when relevant.
- Tests: Unit + integration. Mock heavy external calls (LLM, ASR). High coverage on `src/`.

## Documentation & Context (always fresh)

- [PROJECT_STATUS.md](PROJECT_STATUS.md) — Current feature status, architecture overview, open tasks.
- [AGENT_CONTEXT.md](AGENT_CONTEXT.md) — Complete briefing optimized for LLM context windows.
- [CHANGELOG.md](CHANGELOG.md) & [docs/ROADMAP.md](docs/ROADMAP.md) — History and future plans.
- `docs/` — Architecture, LLM guide, Fas summaries, API docs.

These are kept in sync via `github-project-status` skill after every significant session.

## Security & Privacy

See [SECURITY.md](SECURITY.md). Call center data is sensitive — never commit real customer audio or PII. Use demo/fake data for development.

## When Adding Features

1. Check `docs/LLM_AGENT_GUIDE.md` for patterns.
2. Implement in the right layer (analysis step → registry, LLM provider → src/llm/, dashboard component → components/).
3. Add/update tests.
4. Update relevant docs (ROADMAP, CHANGELOG, AGENT_CONTEXT if major).
5. Re-run `github-project-status` skill.

**Recommendation:** After any change that affects features or architecture, re-run the github-project-status skill so AGENT_CONTEXT.md and PROJECT_STATUS.md stay fresh for the next agent.
