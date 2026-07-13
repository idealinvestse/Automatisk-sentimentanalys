# Project Cleanup Implementation Plan

> **For agentic workers:** Execute task-by-task. Steps use checkbox syntax.

**Goal:** Remove NiceGUI legacy, non-canonical docs, and unused artifacts so the repo matches v0.5 (API + CLI + webui).

**Architecture:** Big-bang delete on `main`; retarget launcher dashboard to Next.js `webui/`.

**Tech Stack:** Python/FastAPI, Next.js webui, Windows launcher, GitHub Actions

---

### Task 1: Delete NiceGUI + NiceGUI-only tests
- [x] Remove `app/archive/`
- [x] Remove NiceGUI-only tests/fixtures/compose

### Task 2: Retarget launcher/install/pyproject/CI away from NiceGUI
- [x] `dashboard_ui=webui`, port 3000
- [x] Drop `dashboard-nicegui` / nicegui deps
- [x] Update CI/Makefile/installer/scripts

### Task 3: Delete docs/agent folders/reports/notebooks
- [x] Canonical docs only + `docs/superpowers/`
- [x] Keep CI baselines in `reports/`
- [x] Remove `.grok/`, `.windsurf/`, `.devin/`, notebooks

### Task 4: Update canonical docs
- [x] README, AGENTS, ROADMAP, CHANGELOG, webui README

### Task 5: Verify pytest + ruff
- [x] Focused suites pass; 811 tests collect; ruff clean on touched paths
