# Analysfunktioner — Implementation Plan (7 ranked ideas)

> **For agentic workers:** Execute units in order U1→U7. Prefer TDD for each unit. Do not invent production corpus data.

**Goal:** Ship the seven ranked ideation directions as coherent pipeline/product changes: shared EvidenceSpan, honest degradation, override provenance, aspect-first surfaces, CCP+routing, quality-OS scaffolding, and a bounded partial-analysis path.

**Architecture:** Extend existing registry + hybrid deep path (`deep_path.py`, `pipeline_steps.py`, `llm/schemas.py`) rather than rewriting the product. YAML profiles remain priors; runtime CCP/routing and honest-unavailable markers make degradation and LLM spend auditable. Aspects become the primary claim unit; call-level sentiment is derived.

**Tech stack:** Python/Pydantic, FastAPI, existing analyzer registry, Next.js webui (minimal AspectCard surface), pytest.

**Origin:** `docs/ideation/2026-07-10-analysfunktioner-ideation.html`, `docs/ANALYZER_STRATEGY.md`

---

## Scope boundaries

**In scope:** Shared EvidenceSpan contract; slim callcenter defaults + unavailable markers; override_provenance on LLM supersession; aspect-first report/API/webui fields; CCP gate before LLM; living analyzer selection; MQM/preference scaffolding (empty corpus OK); partial analysis API + reconciliation hook.

**Out of scope / deferred:** Full streaming-only product rewrite; annotated production corpus; full MQM annotation UI; rewriting all analyzers to mandatory spans; NiceGUI changes.

---

## Implementation units (order)

### U1. EvidenceSpan shared contract (#7)
- **Files:** `src/analysis/evidence.py` (new), `src/llm/schemas.py` (extend + re-export), `src/analysis/aspect.py`, `src/analysis/compliance_risk.py`, tests
- **Decision:** Canonical `EvidenceSpan` gains optional `segment_id`, `start`, `end`; keep `text` as quote; `speaker_role` stays. Helper `make_evidence_span(...)`. Aspect emits `evidence_spans: list[EvidenceSpan]` while keeping string `evidence` for backward compat. Compliance flags include `evidence_spans`.
- **Tests:** `tests/test_evidence_span_contract.py` — construct, aspect output shape, compliance shape, LLM schema still validates.

### U2. Honest degradation (#2)
- **Files:** `configs/analyzer_profiles.yaml`, `src/analysis/deep_path.py`, `src/pipeline.py` / `pipeline_steps.py`, `docs/ANALYZER_STRATEGY.md`
- **Decision:** Slim callcenter `default_selected` to core local + deterministic sensors (sentiment, intent, role, emotion, negation, compliance_risk, customer_effort, active_listening, aspect). Move empathy/trajectory/root_cause/actionable_coaching/insights/etc. to optional. When deep path inactive, skip superseded analyzers even if explicitly listed unless `allow_heuristic_superseded=True`; inject `{status: unavailable, reason: requires_deep_path}` markers into `results`.
- **Tests:** profile slim assertion; unavailable markers when deep path off; heuristics still skippable when deep path on.

### U3. Override provenance (#4)
- **Files:** `src/llm/schemas.py` (`OverrideProvenance`), `src/analysis/deep_path.py` or `src/analysis/provenance.py`, `src/pipeline_steps.py` merge path
- **Decision:** When LLM overwrites local fields (agent_assessment, trajectory, root_cause, refined_aspects), attach `override_provenance` entries with local_source, reason, evidence_spans; log supersession events. Add diversity policy note helper (emotion keyword vs LLM channel).
- **Tests:** provenance attached on merge; log/event structure.

### U4. Aspect-evidence platform (#1)
- **Files:** `src/analysis/aspect_platform.py` (new), `src/pipeline.py` report build, `src/api/schemas.py`, `webui` AspectCard + client types
- **Decision:** Prefer `llm.refined_aspects` over local aspect when present; build `aspect_claims` + `derived_call_sentiment` in results; API `AnalyzerResults` exposes them; AspectCard shows evidence quotes / claim-chart style.
- **Tests:** prefer refined; derive sentiment; API field present.

### U5. CCP + living routing (#5)
- **Files:** `src/analysis/ccp.py` (new), `src/pipeline_steps.py`, optional YAML priors in `configs/`
- **Decision:** Named CCPs before LLM: `pii_clean`, `min_segment_quality`, `sentiment_negation_sanity`. Failed CCP → skip/discard deep output with corrective action logged. `select_analyzers_runtime(profile, features)` adjusts selected set from length/intent/risk; YAML = priors.
- **Tests:** CCP pass/fail; routing adds/removes analyzers.

### U6. Quality OS scaffolding (#3)
- **Files:** `src/quality/` (`mqm.py`, `__init__.py`), `configs/quality_mqm.yaml`, `scripts/evaluate_preference_gate.py`, hook note in `configs/analyzer_eval.yaml` / ANALYZER_STRATEGY
- **Decision:** MQM-like error typology schemas + preference-pair schema; evaluate script exits 0 on empty corpus with clear message; DATA-01 integration points documented — no fake labels.
- **Tests:** schema round-trip; script empty-corpus behavior.

### U7. Partial / streaming path (#6) — bounded
- **Files:** `src/pipeline.py` (`analyze_segments_partial`), `src/api/routers/pipeline.py`, schemas, docs note
- **Decision:** Incremental local analysis on segment window + merge previous results; `reconcile_with_holistic` hook for LLM at hangup/complete. Document full streaming vision in plan/strategy; do not rewrite WS transcription stack.
- **Tests:** partial updates merge; reconcile flag/hook callable.

---

## Verification contract
- Focused pytest for each new test module; all new tests pass.
- No commit unless user asks.
- Update `CHANGELOG.md` [Unreleased] and `ANALYZER_STRATEGY.md` briefly.

## Shipped vs deferred (expected)
| Idea | Expected |
|------|----------|
| #7 EvidenceSpan | Fully used by aspect + compliance + LLM override path |
| #2 Honest degradation | Fully (slim profile + unavailable markers) |
| #4 Provenance | Fully on merge path; diversity = policy helper/notes |
| #1 Aspect platform | Report/API + one webui surface; not full Fas4 search rewrite |
| #5 CCP + routing | Gate + runtime selector; not full HACCP ops UI |
| #3 Quality OS | Scaffold only (schemas/script/CI hook docs) |
| #6 Streaming | Partial API + reconcile hook; full WS-first product deferred |
