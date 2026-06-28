# Analyskvalitetsrapport — Automatisk-sentimentanalys

**Datum:** 2026-06-28  
**Omfattning:** 23 registry-analyzers, 2 LLM-holistic analyzers, 4 post-registry-steg  
**Referens:** [ANALYZER_STRATEGY.md](ANALYZER_STRATEGY.md), [LLM_AGENT_GUIDE.md](LLM_AGENT_GUIDE.md)

---

## Executive summary

Systemet har en **mogen hybridarkitektur**: snabba lokala analyzers via registry-mönstret, selektiv LLM-deep path, och post-processing (agent performance, QA, alerting). Strategin i `ANALYZER_STRATEGY.md` är tydlig och väl implementerad via `deep_path.py`.

| Dimension | Bedömning (1–5) | Kommentar |
|-----------|-----------------|-----------|
| Arkitektur & registry | 5 | Topologisk körning, felisolering, profiler |
| LLM deep path | 5 | Svenska prompts, strikt schema, GDPR-gate |
| Core local analyzers | 3 | Sentiment/intent bra; emotion/role svagare |
| Heuristiska enrichments | 3 | Ojämn kvalitet, schema-gap |
| Dashboard-integration | 3 | Fältnamns-buggar fixade i denna sprint |
| Testtäckning | 3 | Stark infra/LLM; ~15 analyzers utan tester |

**Styrkor:** Registry-infra, LLM-prompts/schemas, intent-klassificering, graceful degradation, llm_judge-testning.

**Huvudproblem (adresserade i denna implementation):** Schema-/fältinkonsistens, dashboard som döljer LLM-data, keyword-only emotion, trajectory utan kundfilter, dialect_sensitivity med false positives.

---

## Arkitektur

```
Audio / Text → Transcription → PII (profil) → Registry analyzers (topologisk)
    → Fas4 enrichment (agent_performance, LLM holistic, QA, alerting)
    → CallAnalysisReport → API / CLI / Dashboard
```

**Deep path:** Aktiveras via `deep_analysis`, `use_mistral_llm`, eller callcenter-profil med ≥6 segment. Hoppar över: empathy, trajectory, insights, root_cause, actionable_coaching.

---

## Inventering per analys

### Tier 1: Core local (callcenter default)

| ID | Avsikt | Kvalitet (1–5) | Kapacitet | Källa |
|----|--------|----------------|-----------|-------|
| `sentiment` | Per-segment sentiment | 4 | HF XLM-R, CPU/GPU | `src/analysis/sentiment.py` |
| `intent` | 10 svenska callcenter-intents | 4 | Heuristic/LoRA | `src/analysis/intent.py` |
| `role` | Agent/kund + Fas4-metriker | 3 | Heuristik + diarization | `src/analysis/role_classifier.py` |
| `emotion` | Multi-label känslor | 3 | Hybrid keyword+sentiment+negation | `src/analysis/emotion.py` |
| `negation` | Svenska negationsmarkörer | 4 | Regler, schema-validerad | `src/analysis/negation.py` |
| `compliance_risk` | Compliance-flaggor | 3 | Regler + agent-rollfilter | `src/analysis/compliance_risk.py` |
| `customer_effort` | CES — friktion | 3 | Keyword/heuristik | `src/analysis/customer_effort.py` |
| `active_listening` | Backchannels, avbrott | 3 | Timing-heuristik | `src/analysis/active_listening.py` |

### Tier 2: Local enrichment (valfria)

| ID | Avsikt | Kvalitet | Superseded av LLM |
|----|--------|----------|-------------------|
| `empathy` | Empati-score 0–100 | 2 | Ja |
| `trajectory` | Sentimenttrend, eskalering | 3 | Ja |
| `insights` | Syntes sentiment/intent/topics | 3 | Ja |
| `root_cause` | Keyword-rotorsaker | 2 | Ja |
| `actionable_coaching` | Regelbaserad coaching | 3 | Ja |
| `upsell_opportunity` | Köpsignaler | 2 | Nej |
| `resolution_probability` | Lösningsgrad | 3 | Nej |
| `predictive` | Churn/eskaleringsrisk | 3 | Nej |
| `summary` | Extractiv sammanfattning | 3 | Nej |
| `topics` | Topic extraction | 3 | Nej |
| `aspect` | ABSA callcenter | 3 | Delvis (LLM refined_aspects) |
| `llm_judge` | Re-bedöm låg-confidence sentiment | 4 | Nej |
| `multi_turn_journey` | Konversationsstadier | 3 | Nej |
| `spoken_normalizer` | ASR-filler-rensning | 2 | Nej |
| `dialect_sensitivity` | Dialekt/slang-flagga | 2 | Nej |

### Tier 3: Deep path (LLM)

| Task | Avsikt | Kvalitet |
|------|--------|----------|
| `trajectory` | Kundens emotionella båge | 5 |
| `refined_aspects` | ABSA med evidens | 5 |
| `root_cause` | Djup rotorsaksanalys | 5 |
| `actionable_summary` | QA-redo sammanfattning | 5 |
| `agent_assessment` | Empati, compliance, coaching | 5 |
| `emotion_trajectory` | Granulära diagram-punkter | 4 |

### Tier 4: Post-registry

| Modul | Avsikt | Kvalitet |
|-------|--------|----------|
| `agent_performance` | Kvantitativa agent/kund-metriker | 4 |
| `compliance_qa` | YAML-scorecard + hybrid LLM | 4 |
| `alerting` | Tröskelbaserade aviseringar | 3 |
| `pii_redactor` | Före LLM-anrop | 4 |

---

## Styrkor

1. **Registry-infra** — Topologisk körning, async, per-analyzer felisolering (`test_analysis_registry.py`).
2. **LLM-lager** — Svenska callcenter-prompts, strikt Pydantic, GDPR-gate för Groq.
3. **Domänanpassning** — Intent-klassificering, aspect-nyckelord, customer_effort för talad svenska.
4. **llm_judge** — Budget, batching, mock-fallback (`test_llm_judge.py`).
5. **Graceful degradation** — Pipeline kraschar aldrig p.g.a. enskild analyzer.

---

## Svagheter (före/efter åtgärd)

| Problem | Status |
|---------|--------|
| Dashboard läste `root_cause.summary` istället för `primary_cause` | **Fixat** |
| Dashboard läste fel fält för QA-rekommendationer | **Fixat** |
| Emotion-timeline visade demo-data | **Fixat** (segment enrichment) |
| Tom-path schema-mismatch (compliance, empathy, CES) | **Fixat** |
| emotion keyword-only trots "hybrid" | **Fixat** |
| trajectory utan kundfilter | **Fixat** |
| intent tuple/dict-inkonsistens | **Fixat** |
| dialect_sensitivity false positives | **Fixat** (omarbetade markörer) |
| aspect duplicerar sentiment | **Fixat** |
| llm_judge utan roll/negation | **Fixat** |
| Few-shot saknas i prompts | **Fixat** |

**Kvarstående (medel/låg prioritet):**
- Intent model-backend som callcenter-default (kräver benchmark mot tränad modell; heuristic-baseline dokumenterad)

---

## Rekommenderade förändringar (implementerade)

### Kategori 1: Kritiska buggar
- Dashboard-fältmappning i `call_detail.py`
- Emotion enrichment i `data_services.py`
- Schema-kompatibla tom-path returns

### Kategori 2: Core kvalitet
- Emotion hybrid (sentiment + negation)
- Trajectory kundfilter + strukturerade eskaleringsobjekt
- Intent dict-output + `intent_utils.py`
- Compliance_risk utökade regler + agent-filter
- Role hälsningsheuristik

### Kategori 3: Konsolidering
- dialect_sensitivity omarbetad
- aspect återanvänder ctx sentiment
- multi_turn_journey tillagd i callcenter optional
- Dokumentation om LLM-supersession i profiler

### Kategori 4: Schema
- Nya schemas: sentiment, emotion, intent, trajectory, role

### Kategori 5: Testning
- `tests/test_analyzer_quality.py` — 8+ analyzers

### Kategori 6: LLM
- llm_judge med roll + negation
- Few-shot-exempel i `prompts.py`
- routing `budget_usd` styr tier

---

## Bedömningsmatris (1–5)

| Analyzer | Avsikt | Kvalitet | Tester | Dashboard |
|----------|--------|----------|--------|-----------|
| sentiment | 5 | 4 | 2 | 4 |
| intent | 5 | 4 | 3 | 4 |
| role | 4 | 3 | 3 | 3 |
| emotion | 4 | 3 | 4 | 4 |
| negation | 4 | 4 | 5 | 2 |
| compliance_risk | 4 | 3 | 4 | 2 |
| customer_effort | 4 | 3 | 4 | 2 |
| active_listening | 4 | 3 | 4 | 2 |
| llm_judge | 4 | 4 | 5 | 4 |
| LLM holistic | 5 | 5 | 5 | 4 |
| dialect_sensitivity | 3 | 2 | 3 | 1 |

*Skala: 1=dålig, 5=utmärkt. Tester/Dashboard uppdaterade efter implementation.*

---

## Prioriterad roadmap (resterande)

Alla punkter från 2026-06-28-planen är nu implementerade:

| Åtgärd | Status |
|--------|--------|
| spoken_normalizer → sentiment/intent | **Klar** — `text_utils.segment_analysis_text`, optional i profil |
| Golden-file-tester | **Klar** — `tests/fixtures/callcenter_golden/`, `tests/test_callcenter_golden.py` |
| Intent benchmark | **Klar** — `scripts/benchmark_intent.py`, `reports/intent_baseline.json` (heuristic 96.3% acc / 20% holdout) |
| Alerting säker parser | **Klar** — token-parser utan `eval()` |
| Strict validation i CI | **Klar** — `ANALYZER_VALIDATION_MODE=strict` i CI + integrationstest |

**Nästa steg (ej blockerande):**
- Train intent **model** via `scripts/train_intent.py` when GPU/training deps available; run `compare_intent_backends.py` before default switch.
- Import anonymized real calls via `scripts/evaluate_real_corpus.py` (outputs to `reports/offline/`, gitignored).
- Dashboard-panel för `compliance_risk` (fortfarande via QA indirekt)

### Post accuracy-program metrics (2026-06-28)

| Analyzer | Val metric | Value |
|----------|------------|-------|
| Intent (heuristic) | macro F1 | 0.765 on `intent_val.jsonl` |
| Intent (heuristic) | accuracy | 0.743 |
| Emotion (labeled fixture) | accuracy | 0.80+ |
| Role (labeled fixture) | accuracy | 1.0 |
| Compliance (labeled fixture) | accuracy | 0.83+ |

---

## Referenser

- [`src/analysis/registry.py`](../src/analysis/registry.py)
- [`configs/analyzer_profiles.yaml`](../configs/analyzer_profiles.yaml)
- [`docs/ANALYZER_STRATEGY.md`](ANALYZER_STRATEGY.md)
