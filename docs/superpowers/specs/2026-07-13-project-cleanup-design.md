# Project cleanup design (v0.5)

**Date:** 2026-07-13  
**Approach:** Big-bang delete (approved)  
**Aggressiveness:** Remove NiceGUI entirely; agent IDE folders; consolidate docs; trim reports/notebooks

## Goal

Leave only code and docs that serve the current v0.5 product surface: FastAPI + CLI + Next.js `webui/` + Windows launcher/installer.

## Delete

- `app/archive/` (NiceGUI dashboard)
- NiceGUI-only tests and fixtures
- `docker-compose.nicegui.yml`
- `.grok/`, `.windsurf/`, `.devin/`
- Non-canonical docs (including `docs/archive/`, plans, FAS/audit reports)
- `notebooks/`
- Non-CI `reports/*` (keep intent/analyzer/finetune/domain baselines)
- Root stubs: `RECOMMENDED_NEXT_TASKS.md`, `ROADMAP.md`, `AGENT_CONTEXT.md`, `PROJECT_STATUS.md`, `docs/CLEANUP_PLAN.md` (after sync)
- Wrapper junk outside inner repo (`debug-*.log`) when present

## Keep (canonical docs)

`README.md`, `AGENTS.md`, `SECURITY.md`, `CHANGELOG.md`, `CONTRIBUTING.md`,  
`docs/ROADMAP.md`, `docs/API.md`, `docs/ARCHITECTURE.md`, `docs/DEVELOPMENT.md`,  
`docs/LLM_AGENT_GUIDE.md`, `docs/PRODUCTION_CHECKLIST.md`, `docs/ANALYZER_STRATEGY.md`,  
`docs/WINDOWS_INSTALL.md`, plus this cleanup spec under `docs/superpowers/`.

## Code updates

- Remove `dashboard-nicegui` extra; drop `nicegui` from `dev`
- CI: no NiceGUI jobs; install without `dashboard-nicegui`
- Launcher/install: dashboard = webui (or clear deprecation); stop importing NiceGUI
- `app/dashboard_launcher.py`: remove or redirect to webui instructions
- Trim `test_analyzer_quality.py` NiceGUI class; update `test_dashboard.py`

## Verification

`pytest -q`, `ruff check src tests launcher`, CI workflow consistent.
