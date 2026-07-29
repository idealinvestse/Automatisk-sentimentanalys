---
title: "feat: Add semantic optional deps for production vector search"
type: feat
date: 2026-06-29
---

# feat: Add semantic optional deps for production vector search

## Summary

Add a `semantic` optional-dependency extra and wire it into install profiles and preflight so production deployments get native vector-search libraries (`sentence-transformers`, `faiss-cpu`, `hdbscan`) instead of slow Python fallbacks.

## Problem Frame

A language-investigation of the codebase found that semantic search and insights aggregation degrade to pure-Python keyword/frequency paths when `sentence-transformers`, `faiss`, and `hdbscan` are missing. These packages are not declared in `pyproject.toml` and no install profile pulls them in, so production installs silently run degraded paths.

## Requirements

- R1. `pyproject.toml` declares a `semantic` optional extra with `sentence-transformers`, `faiss-cpu`, `hdbscan`, and `numpy`.
- R2. `full` and `dev` install profiles include the `semantic` extra via `extras_for_profile`.
- R3. Preflight reports semantic dependency status (informational; missing deps warn but do not fail the overall check).
- R4. `Makefile` exposes `install-semantic` for ad-hoc installs.
- R5. Tests cover the new extra mapping and preflight checks.

## Key Technical Decisions

- **Dedicated `semantic` extra, not `production`:** `API_PRODUCTION` already names API hardening; a separate `semantic` extra avoids naming collision.
- **Include in `full` and `dev` only:** `cli`/`api` profiles stay lean; semantic search is a Fas 4 feature used in full deployments.
- **Preflight is informational:** Missing semantic deps degrade gracefully today; doctor should surface the gap without blocking installs.

## Implementation Units

### U1. Declare semantic extra in pyproject.toml

**Goal:** Make vector-search native libraries installable via pip extras.

**Requirements:** R1

**Dependencies:** none

**Files:** `pyproject.toml`

**Approach:** Add `[project.optional-dependencies] semantic` with pinned minimum versions compatible with existing torch stack.

**Test scenarios:**
- Happy path: `semantic` key exists in pyproject optional-dependencies with expected packages.
- Test expectation: none — config-only; covered by provision test indirectly.

**Verification:** `grep semantic pyproject.toml` shows the extra block.

### U2. Wire semantic into install profiles

**Goal:** Full and dev provision installs pull semantic deps automatically.

**Requirements:** R2

**Dependencies:** U1

**Files:** `src/install/provision.py`, `tests/test_provision.py`

**Approach:** Append `"semantic"` to `InstallProfile.full` and `InstallProfile.dev` lists in `extras_for_profile`.

**Test scenarios:**
- Happy path: `extras_for_profile(InstallProfile.full)` contains `"semantic"`.
- Happy path: `extras_for_profile(InstallProfile.dev)` contains `"semantic"`.
- Edge case: `extras_for_profile(InstallProfile.api)` does not contain `"semantic"`.

**Verification:** `pytest tests/test_provision.py -q` passes.

### U3. Add preflight semantic dependency checks

**Goal:** Doctor command surfaces whether vector search runs in native or fallback mode.

**Requirements:** R3

**Dependencies:** U1

**Files:** `src/install/preflight.py`, `tests/test_preflight.py`

**Approach:** Add `_check_semantic_deps` called for non-minimal profiles; each module check uses existing `_check_import` pattern with `ok=True` and explanatory message when optional.

**Test scenarios:**
- Happy path: preflight report includes checks for `sentence_transformers`, `faiss`, `hdbscan` when profile is `full`.
- Edge case: minimal profile skips semantic checks.

**Verification:** `pytest tests/test_preflight.py -q` passes.

### U4. Makefile install target

**Goal:** Developers can install semantic stack without full profile.

**Requirements:** R4

**Dependencies:** U1

**Files:** `Makefile`

**Approach:** Add `install-semantic` target mirroring existing `install-diarize` pattern.

**Test expectation:** none — Makefile target.

**Verification:** Target exists in `make help` output.

## Scope Boundaries

### Deferred to Follow-Up Work

- Wiring `_embed_texts` / `_cluster_embeddings` into `insights_aggregator.aggregate()` (dead code today).
- Rust/edge inference prototype (`src/edge/`).
- README typo fix on line 32.
