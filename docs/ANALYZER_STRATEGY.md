# Analyzer Strategy

**Purpose:** Define which analyzers run in local vs. deep (LLM) paths. Canonical reference for INSIGHT-02 consolidation.

> **Do not add new thin heuristik-analyzers** without checking overlap with the holistic LLM path (`ConversationMistralAnalyzer` / `GroqAnalyzer`).

## Tiers

| Tier | Analyzers | When |
|------|-----------|------|
| **Core local** | `sentiment`, `intent`, `role`, `negation`, `compliance_risk` | Always (fast, offline-capable) |
| **Local enrichment** | `customer_effort`, `active_listening`, `resolution_probability`, `upsell_opportunity`, `predictive` | When `deep_analysis=false` or LLM unavailable |
| **LLM superseded** | `empathy`, `trajectory`, `insights`, `root_cause`, `actionable_coaching` | Skipped when deep path active (`should_use_any_llm`) |
| **Deep path** | Holistic LLM (`results["llm"]`) + optional `llm_judge` | `deep_analysis`, `use_mistral_llm`, or callcenter profile with ≥6 segments |

## Deep path decision

Implemented in `src/pipeline_steps.py`:

- `should_use_any_llm()` — profile + segment count + explicit flags
- When true, `run_registry_analyzers()` receives `skip_llm_superseded=True`
- LLM output merges into report; dashboard prefers LLM trajectory/empathy when present

## Profile defaults (`configs/analyzer_profiles.yaml`)

**callcenter `default_selected`:** sentiment, intent, role, emotion, negation, compliance_risk, customer_effort, active_listening

**callcenter `optional`:** empathy, insights, trajectory, llm_judge, upsell_opportunity, resolution_probability, root_cause, predictive, actionable_coaching, multi_turn_journey, spoken_normalizer (ASR filler cleanup; feeds sentiment/intent when enabled)

When LLM runs, superseded optional analyzers are skipped automatically. Without LLM, enable them explicitly via `selected_analyzers` or profile config.

## Overlap matrix

| Local analyzer | LLM equivalent | Merge rule |
|----------------|----------------|------------|
| `trajectory` | `llm.trajectory`, `emotion_trajectory` | Dashboard prefers LLM |
| `empathy` | `agent_assessment.empathy_score` | LLM overwrites when set |
| `insights` | `actionable_summary`, `root_cause` | LLM preferred |
| `root_cause` | `llm.root_cause` | Skip local when deep path |
| `actionable_coaching` | `agent_assessment` coaching fields | Skip local when deep path |
| `aspect` (optional) | `refined_aspects` | LLM preferred when both run |

## Adding new analysis

1. Prefer extending holistic LLM tasks in `src/llm/mistral_analyzer.py` for reasoning-heavy features
2. Use registry analyzers only for fast, deterministic, offline signals
3. Register in `configs/analyzer_profiles.yaml` under `optional` first; promote to `default_selected` after evaluation
4. Run `sentimentanalys new-analyzer` for boilerplate

## Historical note

Replaces `docs/PROPOSED_ANALYZERS.md` (2026-06-27 research list). New analyzer proposals should justify why LLM path is insufficient.
