# Analyzer Strategy

**Purpose:** Define which analyzers run in local vs. deep (LLM) paths. Canonical reference for INSIGHT-02 consolidation and 2026-07-10 analysfunktioner directions.

> **Do not add new thin heuristik-analyzers** without checking overlap with the holistic LLM path (`ConversationMistralAnalyzer` / `GroqAnalyzer`).
> **Null före bluff:** superseded fields are `status: unavailable` when deep path is off — not quality-2 heuristics.

## Tiers

| Tier | Analyzers | When |
|------|-----------|------|
| **Core local** | `sentiment`, `intent`, `role`, `emotion`, `negation`, `compliance_risk` | Always (fast, offline-capable) |
| **Deterministic sensors** | `customer_effort`, `active_listening`, `aspect` | Default callcenter slim profile |
| **Local enrichment (optional)** | `summary`, `topics`, `resolution_probability`, `upsell_opportunity`, `predictive`, `multi_turn_journey` | Explicit select / living routing |
| **LLM superseded** | `empathy`, `trajectory`, `insights`, `root_cause`, `actionable_coaching` | Skipped locally by default; filled by deep path or marked unavailable |
| **Deep path** | Holistic LLM (`results["llm"]`) + optional `llm_judge` | `deep_analysis`, `use_mistral_llm`, or callcenter profile with ≥6 segments **and** CCP pass |

## Deep path decision

Implemented in `src/pipeline_steps.py`:

- `should_use_any_llm()` — profile + segment count + explicit flags
- **CCP gate** (`src/analysis/ccp.py`): `pii_clean`, `min_segment_quality` (≥6 segments, avg ≥12 chars), `sentiment_negation_sanity` — failed CCP blocks LLM
- **WebUI trust surface**: `TrustSurfaceCard` shows `degradation`, `deep_path_ccp`, `analyzer_routing`, `override_provenance` (Call Detail, Testlabb, Analysis)
- **Redis transcription hub**: `TranscriptionEventHub` publishes to `ws:transcription:events` when Redis available; in-memory fallback
- **EvidenceSpan (core)**: emotion + negation emit spans; aspect/compliance already covered
- **Edge MVP**: offline sentiment + intent + negation + aspect (`src/edge/local_inference.py`)
- **Transcription hook**: `run_partial_analysis` on `POST /transcribe` runs incremental partial path after ASR
- When LLM runs, superseded locals stay skipped; `override_provenance` records supersessions
- Living routing (`select_analyzers_runtime` in `CallAnalysisPipeline._run_local_analysis`): two-pass — segment-count trim/expand first, then extras from intent/risk; YAML = priors; `results["analyzer_routing"].applied=True`

## Honest degradation

- Default: do not run LLM-superseded analyzers unless `allow_heuristic_superseded=True`
- `inject_unavailable_markers()` writes `{status: unavailable, reason: requires_deep_path}` for empathy/trajectory/insights/root_cause/actionable_coaching when LLM did not produce them
- Dashboard/API should show “kräver deep path” for unavailable fields

## Aspect-evidence platform

- Primary product unit: `results["aspect_claims"]` (prefer `llm.refined_aspects`, else local ABSA)
- Secondary: `results["derived_call_sentiment"]` aggregated from aspect claims
- Shared `EvidenceSpan` (`src/analysis/evidence.py` / `src/llm/schemas.py`) on aspect + compliance + LLM overrides

## Profile defaults (`configs/analyzer_profiles.yaml`)

**callcenter `default_selected`:** sentiment, intent, role, emotion, negation, compliance_risk, customer_effort, active_listening, aspect

**callcenter `optional`:** summary, topics, resolution_probability, predictive, multi_turn_journey, empathy, trajectory, root_cause, actionable_coaching, insights, llm_judge, upsell_opportunity, spoken_normalizer, dialect_sensitivity

## Intent backend selection (DATA-01)

- **Default:** `heuristic` in `IntentAnalyzer` (phrase boosts + disambiguation rules).
- **Benchmark:** `scripts/benchmark_intent.py --val-file data/intent_val.jsonl` (macro F1 primary).
- **Model A/B:** `scripts/compare_intent_backends.py`; switch to `model` only if macro F1 ≥ heuristic + 0.05 (`configs/analyzer_eval.yaml`).
- **Quality OS:** MQM + preference gate scaffolding in `src/quality/` + `configs/quality_mqm.yaml` + `scripts/evaluate_preference_gate.py` (empty corpus = CI skip; wire real labels via DATA-01).

## Overlap matrix

| Local analyzer | LLM equivalent | Merge rule |
|----------------|----------------|------------|
| `trajectory` | `llm.trajectory`, `emotion_trajectory` | LLM + override_provenance |
| `empathy` | `agent_assessment.empathy_score` | LLM overwrites when set |
| `insights` | `actionable_summary`, `root_cause` | LLM preferred |
| `root_cause` | `llm.root_cause` | Skip local when deep path |
| `actionable_coaching` | `agent_assessment` coaching fields | Skip local when deep path |
| `aspect` (sensor) | `refined_aspects` | Prefer refined → `aspect_claims` |

## Partial / streaming path

- `CallAnalysisPipeline.analyze_segments_partial` + `POST /analyze_pipeline/partial`
- Incremental local merge; `reconcile=True` runs Fas4/LLM holistic reconciliation
- Full WS-first product rewrite deferred (see implementation plan)

## Adding new analysis

1. Prefer extending holistic LLM tasks in `src/llm/mistral_analyzer.py` for reasoning-heavy features
2. Use registry analyzers only for fast, deterministic, offline signals with `EvidenceSpan` where claims are made
3. Register in `configs/analyzer_profiles.yaml` under `optional` first; promote to `default_selected` after evaluation
4. Run `sentimentanalys new-analyzer` for boilerplate

## Historical note

Replaces `docs/PROPOSED_ANALYZERS.md` (2026-06-27 research list). Ideation 2026-07-10: `docs/ideation/2026-07-10-analysfunktioner-ideation.html` → plan `docs/plans/2026-07-10-analysfunktioner-implementation-plan.md`.
