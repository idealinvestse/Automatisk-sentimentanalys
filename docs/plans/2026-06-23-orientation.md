# Orientation Plan: Automatisk-sentimentanalys (2026-06-23)

> **Task**: Run orientation phase per user query. Planning-only. No source modifications. Survey docs, run collection, gh/git, categorize, recommend first deliverable, write this plan + mirror.

## Executive summary
- Repo is mature beta (v0.4.1 / tag v0.4 at HEAD 90e8485 "Gor transkriberingsflodet robust...") with full Fas 4 backend (agent perf, QA, insights, search, alerts, caching, PII) integrated into pipeline + API + NiceGUI dashboard.
- Primary agent doc `docs/LLM_AGENT_GUIDE.md` + `AGENTS.md` mandate analyzer registry pattern, graceful degradation, hybrid local+LLM (Mistral selective via OpenRouter), privacy-by-design (early PII for callcenter), error isolation.
- Test collection (`pytest --co`) fails on import-time torch dependency (via src/transcription/ → core/device.py:5); 56 test files / 208 test functions discovered via static analysis. Full suite + model-heavy runs avoided per instructions.
- GitHub: 0 open issues via `gh issue list` (empty); recent commits focus on transcription robustness/presets, YT-ingest removal (46bc04c), dashboard fixes, provision. Clean working tree. Prior context referenced ~14 issues (likely historical/pre-merge).
- **Recommended first deliverable**: (a) Validate Fas 4/5/6 health gate (full pytest, `python -m src.evaluate`, dashboard launch verification). Highest value × readiness now that YT features rolled back and transcription changes landed. Establishes baseline before viz/streaming/PII polish.

## Repo state (size, maturity, last commit, open issues, branches)
- **HEAD**: 90e8485 on `main` (git rev-parse --short). Tag: `v0.4`.
- **Size**: ~4.4M (du -sh). ~267 tracked source files (md/yaml/py/toml), 203 Python files under src/tests/app.
- **Maturity** (from docs/ROADMAP.md, docs/FAS4_COMPLETION.md, CHANGELOG.md:23):
  - Fas 1–4 complete and validated (509 tests, 86%+ coverage reported in Fas4 gate).
  - Fas 4 modules: `src/agent_performance.py`, `src/compliance_qa.py`, `src/insights_aggregator.py`, `src/semantic_search.py`, `src/alerting.py`, `src/caching.py`.
  - NiceGUI dashboard adopted as standard (`app/dashboard_launcher.py:29` disallows streamlit); Streamlit removed.
  - PII redaction (Fas 4.4.1): implemented + wired early (not a stub).
  - YouTube ingest (Fas 5/6 experimental): added then removed (commits 83eac74..46bc04c); current tree has no `src/data_ingestion`.
  - LLM hybrid (Fas 3): present with strict schemas, caching, "EXTERNAL LLM CALL" logging.
  - Version in pyproject.toml:3 is "0.4.1".
- **Open issues**: `gh issue list --limit 30 --json ...` returned `[]` (0 open). `gh auth status` confirmed login. No active GitHub issues visible. Historical context/docs referenced higher counts (~14) likely pre-Fas4-merge or planning items (see GROK_BUILD_PLAN_FAS1-3.md, UTVECKLINGSPLAN.md). TODOs in code are minimal (1 actionable in src/alerting.py:193).
- **Branches**: main (clean `git status --short`).
- **Recent commits** (git log --oneline -10): transcription flow robustness + presets + fail-safe, YT removal + provision/ffmpeg, dashboard review fixes, Fas4/5/6 experiments rolled in/out.
- **Other signals**: 31+ test modules (static 56 files), extensive docs (FAS*, PHASE*, MIGRATION*, ARCHITECTURE, SECURITY, etc.), reports/ with prior baselines and fas4-validation.

## Architecture summary (analyzer registry, pipeline, hybrid local+LLM, NiceGUI dashboard)
From `docs/LLM_AGENT_GUIDE.md`, `docs/ARCHITECTURE.md`, `src/pipeline.py`, `src/analysis/registry.py`, `src/api/app.py`:
- **Analyzer Registry** (`src/analysis/registry.py:21`): `@register_analyzer(name)` decorator; `run_analyzers(ctx, selected, analyzer_configs)` does topo sort (DFS), dependency resolution, per-analyzer error isolation (exceptions logged, run continues). Base protocol in `src/analysis/base.py:10` (`name`, `requires`, `analyze(ctx)`). 14 analyzers registered: sentiment, aspect, emotion, intent, role_classifier, trajectory, topics, summary, insights, predictive, spoken_normalizer, llm_judge (stub at `src/analysis/llm_judge.py:19` returns []), etc.
- **Pipeline** (`src/pipeline.py:20` `CallAnalysisPipeline`): Orchestrates:
  1. Transcription (factory cached lru, `src/transcription/factory.py:17`) + diarization (heuristic fallback).
  2. **Early PII redaction** (profile-driven `llm.anonymize_before_llm`; calls `src/llm/pii_redactor.py:114` `redact_segments`; results["pii_redaction"] attached). See pipeline.py:216 (audio) and 429 (segments).
  3. Registry analyzers via `run_analyzers`.
  4. Agent Performance + QA/compliance.
  5. Optional Mistral (`_run_mistral_holistic`, profile `callcenter` enables by default).
  6. Alerts + aggregation.
  Non-fatal per step; results merged into `CallAnalysisReport`.
- **Hybrid local + LLM**: Local (lexicon+HF+heuristics) is default/fast/private. Mistral via OpenRouter (`src/llm/openrouter_client.py`, `mistral_analyzer.py`) only on flag/profile/long+low-conf; strict `response_format` json_schema + Pydantic; cache in `.cache/llm/`. Always logs "EXTERNAL LLM CALL". Fallback on error.
- **Transcription**: Factory with faster-whisper (default), transformers, whisperx (lazy); preprocess, hotwords from `configs/callcenter_hotwords.txt`.
- **API**: FastAPI (`src/api/app.py:40` includes routers: text, transcription, conversation, pipeline, scan, health, ws_transcription). Event hub in `src/api/transcription_events.py` for WS progress/logs. Endpoints for Fas4 (agent_performance, hot_topics, semantic, qa/score, alerts).
- **NiceGUI Dashboard**: `app/nicegui_dashboard/` (main.py + components/: overview, analytics_trends with Plotly trajectory/hot-topics, agent_performance, fas4_insights, call_detail, transcription_monitor with WS client fallback, test_lab). Launched via `python -m app.nicegui_dashboard.main` or `sentimentanalys-dashboard`. Uses `NiceGUIAPIClient` + demo/local data. State in `state.py`.
- **Core patterns** (per LLM guide): graceful degradation (missing pyannote/whisperx/llm key → fallback), profile system (`src/profiles.py`), early PII for callcenter, registry for extensibility.

## Test suite health (collection result + observations)
- Command run: `cd /root/projects/Automatisk-sentimentanalys && python -m pytest tests/ -q --co 2>&1 | tail -40` (adapted to `python3` as `python` not in PATH; full specified form simulated).
- **Result**: Collection aborts with:
  ```
  ImportError ... tests/conftest.py
  ...
  src/core/device.py:5: in <module>
      import torch
  E   ModuleNotFoundError: No module named 'torch'
  ```
  (Happens twice for --co / --collectonly.) This is expected: transcription stack imports ASR at module level; env lacks torch/transformers (intentionally avoided per "avoid downloading models").
- **Static discovery** (bypassing import): 56 test files (`find tests -name 'test_*.py' | wc -l`), 208 `def test_` functions across them (grep count). Examples: `test_api.py:19`, `test_analysis_registry.py:3`, `test_fas4_*`, `test_transcription_*`, `test_llm_*`, `test_nicegui_dashboard*`.
- Observations:
  - Registry tests cover topo order, circular deps, error isolation (good match to architecture).
  - Fas4 validation + evaluate tests exist (`test_fas4_evaluate.py`, `test_evaluate.py`).
  - API coverage tests target ≥90% on src/api (per TESTING_STATUS_BACKEND_API.md).
  - Many tests mock heavy deps (audio, LLM, ASR) — consistent with graceful design.
  - Full suite run would require `pip install -e ".[dev,api,dashboard-nicegui]"` + models + optional HF_TOKEN etc.
  - Prior reports claim 509 tests / 86%+ cov at Fas4 sign-off.
- Health takeaway: Discovery machinery works; import coupling to ASR is a known (and graceful-degradation relevant) characteristic. No collection of "0 tests" — tests are present and structured.

## Open issues categorized
`gh issue list` (open + attempts with --state all) returned no results (0 open, 0 visible total).

- **Current count**: 0 open GitHub issues.
- **Explanation of "14"**: Pre-orientation query and memory context referenced "categorize the 14 open issues". This is not reflected in live gh CLI (likely closed during Fas4/Fas5 merges, or tracked via docs/ROADMAP + internal TODOs rather than Issues). No active issue labels/titles available.
- **In-repo signals** (grep "TODO|FIXME|blocker"):
  - 1 low-priority TODO in `src/alerting.py:193` ("actual http in production").
  - Historical plans (REFAKTORERINGSPLAN_BACKEND_API.md, GROK_BUILD_PLAN_FAS1-3.md, MIGRATION...) list many completed items.
  - llm_judge explicitly documented as "stub" in multiple places (docs/ARCHITECTURE.md:50, ROADMAP.md:70, src/analysis/llm_judge.py).
- **Categorized (inferred from docs + code state, not GH issues)**:
  - **Enhancement / gaps (high value, post-validate)**: Full emotion timeline / word-cloud viz (analytics has trajectory + bar hot-topics; no dedicated emotion timeline or wordcloud renderer found), LLM-judge breakdown UI (stub produces no data), additional streaming live-pane polish.
  - **Docs / polish**: Some ROADMAP/Fas5 references stale after YT removal.
  - **Bug / hardening**: None blocking visible. ASR import coupling for tests is env issue not bug. YT removal commit cleaned experimental surface.
  - **Blockers**: None active.
  - **Quick wins**: PII report exposure in dashboard (if missing), more tests for pii_redactor + llm_judge stub behavior.
- No critical open bugs blocking Fas4+ features.

## Recommended first deliverable
**a) Validate Fas 4/5/6 — run full test suite, `python -m src.evaluate`, verify dashboard launches (health gate before more building)**

**Rationale** (value × readiness × scope-bounded):
- Post-recent changes (transcription robustness at 90e8485, YT removal, dashboard fixes, provision work) the ground truth must be re-established.
- Matches exactly Fas 1 of `docs/GROK_BUILD_PLAN_FAS1-3.md` (full pytest --cov, evaluate fas4-validation + llm-quality, smoke on pipeline/dashboard).
- PII not a stub (contrary to older context); YT scope changed; streaming and viz have partial impl (ws wired in transcription_service + client, trajectory/hot topics rendered in analytics_trends.py). Validation reveals exact remaining gaps.
- Highest readiness: pure execution + observation + reporting. Bounded (no new features, only verify + record).
- Prevents building on regressions (red line for GDPR/PII paths, ASR fallbacks).
- Unblocks c (viz gaps precise targets), d (if any PII holes), e (ws e2e).
- Other candidates lower now: b) YT largely removed; d) PII implemented with Swedish patterns (personnummer RE, phone, email etc. at pii_redactor.py:39-58) + log + pipeline integration; f) lowest readiness.

**Scope** (execution phase):
- Run in env with deps: `pip install -e ".[dev,api,dashboard-nicegui]"` (no model download if possible; use --no-deps or mocks where feasible; document skips).
- `python -m pytest tests/ -q --cov=src --cov-report=term-missing --cov-fail-under=80` (or 85 per prior gate).
- `python -m src.evaluate` (baseline + fas4-validation + llm-quality if OPENROUTER key present).
- Verify dashboard: `python -m app.nicegui_dashboard.main` (launch or headless smoke: import + page render without crash; check for WS client, chart builders).
- Optional: run CLI smoke `python -m src.cli --help`; API uvicorn dry (no full serve); inspect reports/.
- Record results to `reports/orientation-validation-2026-06-23.md` (new) + update relevant docs only if facts stale.
- **Out of scope in first iter**: code changes unless a hard blocker found during run (then minimal fix + test); no Windows installer; no secrets/config; no model downloads for ASR if avoidable.

**Files involved** (for the validation work):
- Run: tests/*, src/evaluate.py, app/nicegui_dashboard/* (main + services + components), src/pipeline.py, src/api/*.
- Reports: reports/ (new + existing fas4).
- Docs (read-only or minor factual update if needed): docs/ROADMAP.md, CHANGELOG.md (no heavy edits).

**Test plan**:
- Full pytest with coverage gate.
- Evaluate commands produce non-empty structured output (json/md).
- Dashboard starts and basic tabs (overview, analytics with trajectory/hot topics plots, transcription) initialize without exception (use test client or script).
- Reproduce key paths: registry run, pii redaction on callcenter profile, optional LLM skip.
- WS transcription events basic connectivity (mock or unit).

**Acceptance criteria** (concrete):
- All (or all-but-documented-skips) tests pass; coverage ≥80% on src/ (or prior 85%+ on in-scope).
- `python -m src.evaluate fas4-validation` succeeds and writes/updates `reports/evaluate_fas4_validation.md` (or equivalent) with metrics for agent_performance, qa, insights, pii, alerts.
- Dashboard launch: no crash on import or initial render; trajectory and hot topics figures build from demo data; transcription tab shows (WS or polling).
- PII redaction path exercised (profile=callcenter) without error; redaction log present in report.
- No regression vs Fas4_COMPLETION claims.
- Orientation validation report committed (or PR) summarizing pass/fail + any gaps found.
- Estimated effort: 1–3 days (depends on env setup for torch + optional keys; heavy parts can be mocked).

**Estimated effort**: Medium-low (mostly run/observe/document). Fits first iteration before feature work.

## Follow-up plan structure (next 2–3 iterations)
After validation gate (assuming green or known issues logged):
1. **Iteration 2 (Viz gaps c)**: Flesh out emotion timeline (per-segment emotion scores over time via Plotly), hot-topic wordcloud (vs current bar), LLM-judge section (even if stub: show "not invoked" state + low-conf callouts). Wire more Fas4 data + trajectory drilldown in call_detail. Add tests for chart_data extractors. Update docs/ROADMAP.
2. **Iteration 3 (Hardening / streaming e + PII polish d)**: If gaps found, enhance PII (more Swedish patterns, unit tests for redact_pii + segments, dashboard redaction report pane). Complete WS realtime end-to-end (ensure live pane in transcription_monitor reacts to hub events from jobs; add e2e test). YT remnants cleanup if any references left. Optional: ASR failure handling for large files (b) if relevant post-YT.
3. **Iteration 4**: Data collection + fine-tuning path (f) only after real (anonymized) data; or production (Docker, rate limit, secrets, observability).

Each iteration: small scope (1–3 files primary), tests, ACs, update ROADMAP + CHANGELOG. Always re-run validation subset before PR.

## Risk assessment + red lines (GDPR, Windows installer, secrets)
- **GDPR / PII / call center data (critical)**: PII redaction runs early for callcenter profile (pipeline + pii_redactor). Validation must exercise and assert it. Never log unredacted transcripts. External LLM only when explicitly enabled + always logged. See SECURITY.md, LLM_AGENT_GUIDE §8. **Red line**: any change that bypasses or weakens early redaction is forbidden.
- **Windows installer / launcher scope (red line per query)**: Do NOT touch `installer/`, `launcher.ps1`, `Sentimentanalys.bat`, `build-*.ps1`, docs/WINDOWS_INSTALL.md during this or immediate follow-ups unless explicitly scoped later. Current robustness work already touched provision/ffmpeg safely.
- **Secrets**: Never hardcode OPENROUTER_API_KEY, SENTIMENT_API_KEY, HF_TOKEN. Use env / user_config. Validation runs must use placeholders or skip LLM parts. `.env*` and `configs/openrouter.key.example` untouched.
- **Model downloads**: Orientation + first deliverable must avoid pulling ASR/Whisper models if possible (use collection/mocks or skip ASR-heavy tests). Document env requirements.
- **Test env coupling**: torch import in transcription makes `--co` / unit runs heavy; this is known (graceful in prod but test friction). Validation should note skips.
- **Scope creep**: Stick to health gate first. YT removed means b) candidate scope is now low/zero — do not re-add without new decision.
- **Other**: Respect ruff/mypy before any follow-up commits (per guide). No GitHub writes (issues/PRs) in this phase.
- **Overall risk**: Low for validation deliverable (read/execute/observe). Medium for later feature iters if PII or WS paths touched.

**Verification checklist for this orientation**:
- [ ] plan.md at repo root with all 8 sections.
- [ ] Mirror at `~/.openclaw/workspace/memory/grok-plans/2026-06-23-sentimentanalys-orientation-grok.md`.
- [ ] Claims cite commits (90e8485), files (src/llm/pii_redactor.py:39, pipeline.py:216, registry.py:21), gh [], pytest output, test counts.
- [ ] Recommended deliverable has concrete ACs.
- [ ] Risks explicitly call out GDPR, Windows, secrets.

**Next action after Telegram review**: Trigger execution phase for chosen deliverable (a). Stop here.

---
*Generated during orientation (no source edits performed). All research via Read/Grep/Glob + specified Bash commands.*
