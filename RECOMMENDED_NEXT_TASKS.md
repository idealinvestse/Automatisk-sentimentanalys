# Rekommenderade uppgifter

**Uppdaterad:** 2026-06-28  
**Backlog:** [docs/CLEANUP_PLAN.md](docs/CLEANUP_PLAN.md) (Fas 0–4 i stort sett klara)  
**Edge AI synergier:** [SYNERGY_ANALYSIS_Fas5.md](SYNERGY_ANALYSIS_Fas5.md)

## Högsta prioritet (v0.5)

1. **PROD-01** Observability: strukturerad loggning, Prometheus metrics, tracing för långa samtal
2. **DATA-01** Fine-tuning pipeline + domändata (Fas 2)
3. **INSIGHT-02** Djupare holistisk LLM-insikt och coaching (färre, bättre analyzers)

## Medel prioritet

4. **PIPE-01** Flytta LLM-routing (`_run_mistral_holistic` / `_run_groq_holistic`) till `pipeline_steps.py` om `pipeline.py` ska under ~500 rader
5. **DEPS-01** Avveckla `requirements*.txt` helt; endast `pyproject.toml` optional-deps
6. **EDGE-01** Edge AI + model catalog synergier (se `SYNERGY_ANALYSIS_Fas5.md`)

## Klart (2026-06-28 städning)

| Uppgift | Leverans |
|---------|----------|
| DOC-01 | `docs/CLEANUP_PLAN.md`, `docs/archive/`, canonical docs |
| EXT-01 | `sentimentanalys new-analyzer` + `templates/new_analyzer_template.py` |
| REF-02 | Streamlit borttagen, NiceGUI enda dashboard-väg |
| REF-03 | Pipeline: `_run_local_analysis`, `_run_fas4_enrichment`, `_build_report` |
| INSIGHT-01 | empathy, CES, actionable_coaching (fanns redan) |
| predictive | Adapter → `RiskAnalyzer` |
| llm_judge | Implementerad + meta-bugg fixad |
