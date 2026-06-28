# Projektstatus - Automatisk Sentimentanalys

**Senast uppdaterad:** 2026-06-28  
**Version:** 0.4.1 (v0.5-prep)

> **Canonical roadmap:** [docs/ROADMAP.md](docs/ROADMAP.md)  
> **Städplan:** [docs/CLEANUP_PLAN.md](docs/CLEANUP_PLAN.md) — Fas 0–4 i stort sett klara  
> **Produktion:** [docs/PRODUCTION_CHECKLIST.md](docs/PRODUCTION_CHECKLIST.md)  
> **Full agent-briefing:** [AGENT_CONTEXT.md](AGENT_CONTEXT.md)

## Nuvarande läge

| Område | Status |
|--------|--------|
| Core pipeline + registry | ✅ `pipeline.py` ~450 LOC; Fas-4/LLM i `pipeline_steps.py` (PIPE-01) |
| Beroenden | ✅ Endast `pyproject.toml` optional-deps (DEPS-01) |
| API (FastAPI) | ✅ Fas 4 sign-off + `/metrics` (OBS-01) |
| NiceGUI dashboard | ✅ Standard-UI (Streamlit avvecklad) |
| Groq + OpenRouter LLM | ✅ Dynamisk pricing via `model_catalog` |
| Edge AI (Fas 5) | 🟡 Grundstruktur (`src/edge/`) |
| LLM model catalog | ✅ CLI + dashboard Test Lab |
| Grok Build skills | ✅ `.grok/skills/` (6 custom skills) |
| Analyzer DX | ✅ `sentimentanalys new-analyzer` + mall |
| Dokumentation | ✅ Konsoliderad + produktionschecklista |
| CI / tester | ✅ Kör `pytest -q` för live antal |

## Kvar (medveten skuld)

- PROD-01: distribuerad tracing, HTTP request metrics, strukturerad JSON-loggning
- Full production fine-tuning training loop
- Edge AI network + Fas 6 commercialization (se `SYNERGY_ANALYSIS_Fas5.md`)

## Nästa prioriteringar (v0.5)

1. Fine-tuning pipeline + domändata (Fas 2)
2. PROD-01 återstående: tracing + request latency histograms
3. Förbättra holistisk LLM-insikt och coaching (inte fler tunna regelanalyzers)
4. Model picker + cost-optimized routing baserat på catalog
