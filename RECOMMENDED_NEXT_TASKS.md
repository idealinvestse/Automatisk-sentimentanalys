# Rekommenderade uppgifter

**Uppdaterad:** 2026-06-28  
**Audit:** [docs/PROJECT_AUDIT_2026-06-28_FULL_REVIEW.md](docs/PROJECT_AUDIT_2026-06-28_FULL_REVIEW.md)  
**Backlog:** [docs/CLEANUP_PLAN.md](docs/CLEANUP_PLAN.md)  
**Produktion:** [docs/PRODUCTION_CHECKLIST.md](docs/PRODUCTION_CHECKLIST.md)  
**Edge AI synergier:** [SYNERGY_ANALYSIS_Fas5.md](SYNERGY_ANALYSIS_Fas5.md)

## Kategoriserad backlog (2026-06-28)

Prioritering efter genomförd audit + topp-10 förbättringar.

### A — Analyzers & kvalitet

| ID | Uppgift | Status |
|----|---------|--------|
| A1 | Bugfixar (aspect, coaching deps, role/sentiment, journey, emotion) | ✅ Klart |
| A2 | Enhetstester heuristik-analyzers | ✅ Klart (`tests/test_heuristic_analyzers.py`) |
| A3 | Delad transcript-byggare Mistral/Groq | ✅ Klart (`src/llm/transcript_utils.py`) |
| A4 | Städa orphan `insights_aggregator_v2` + template-import | ✅ Klart |
| A5 | Negation schema-validering (listor per item) | ✅ Klart |

### B — QA, coaching & insikter

| ID | Uppgift | Status |
|----|---------|--------|
| B1 | `local_signals` → QA-scoring (empati/compliance) | ✅ Klart |
| B2 | Dokumentera holistisk LLM dual-path | ✅ Klart (`docs/LLM_AGENT_GUIDE.md` §5.3) |
| B4 | InsightsAggregator sentiment-trend + tester | ✅ Klart |

### C — Observability & produktion

| ID | Uppgift | Status |
|----|---------|--------|
| C2 | HTTP request metrics middleware | ✅ Klart (`http_requests_total`, latency histogram) |
| PROD-01 | Tracing + full HTTP metrics dashboard | 🔄 Delvis — `/metrics` + checklista kvar att utöka |

### D — Data & modeller (nästa våg)

| ID | Uppgift | Status |
|----|---------|--------|
| DATA-01 | Fine-tuning pipeline + domändata | ⏳ Planerad |
| INSIGHT-02 | Djupare holistisk LLM-insikt (färre, bättre analyzers) | ⏳ Planerad |

### E — Edge & dokumentation

| ID | Uppgift | Status |
|----|---------|--------|
| EDGE-01 | Edge AI + model catalog synergier | ⏳ Se `SYNERGY_ANALYSIS_Fas5.md` |
| E4 | Kategoriserad backlog (denna sektion) | ✅ Klart |

## Högsta prioritet (v0.5)

1. **PROD-01** Observability: HTTP metrics ✅; tracing + dashboards kvar
2. **DATA-01** Fine-tuning pipeline + domändata (Fas 2)
3. **INSIGHT-02** Djupare holistisk LLM-insikt och coaching (färre, bättre analyzers)

## Medel prioritet

4. **EDGE-01** Edge AI + model catalog synergier (se `SYNERGY_ANALYSIS_Fas5.md`)

## Klart (2026-06-28 audit)

| Uppgift | Leverans |
|---------|----------|
| DEPS-01 | `requirements*.txt` borttagna; `pyproject.toml` optional-deps + profiler |
| PIPE-01 | Fas-4 enrichment + LLM routing i `pipeline_steps.py`; `pipeline.py` < 550 LOC |
| DOC-02 | `docs/PRODUCTION_CHECKLIST.md` |
| OBS-01 | `GET /metrics`, Prometheus gauges, `AlertingStateManager` sync |
| DOC-01 | `docs/CLEANUP_PLAN.md`, `docs/archive/`, canonical docs |
| EXT-01 | `sentimentanalys new-analyzer` + mall |
| REF-02 | Streamlit borttagen, NiceGUI enda dashboard-väg |
| REF-03 | Pipeline: `_run_local_analysis`, delad Fas-4 post-processing |
| INSIGHT-01 | empathy, CES, actionable_coaching (fanns redan) |
| predictive | Adapter → `RiskAnalyzer` |
| llm_judge | Implementerad + meta-bugg fixad |
