# Rekommenderade uppgifter

**Uppdaterad:** 2026-06-28  
**Audit:** [docs/PROJECT_AUDIT_2026-06-28_FULL_REVIEW.md](docs/PROJECT_AUDIT_2026-06-28_FULL_REVIEW.md)  
**Backlog:** [docs/CLEANUP_PLAN.md](docs/CLEANUP_PLAN.md)  
**Produktion:** [docs/PRODUCTION_CHECKLIST.md](docs/PRODUCTION_CHECKLIST.md)  
**Edge AI synergier:** [SYNERGY_ANALYSIS_Fas5.md](SYNERGY_ANALYSIS_Fas5.md)

## Högsta prioritet (v0.5)

1. **PROD-01** Observability (delvis klart): tracing + HTTP metrics kvar; `/metrics` + checklista ✅
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
