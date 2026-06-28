# Projektstatus - Automatisk Sentimentanalys

**Senast uppdaterad:** 2026-06-28  
**Version:** 0.4.1 (v0.5-prep)

> **Canonical roadmap:** [docs/ROADMAP.md](docs/ROADMAP.md)  
> **Städplan:** [docs/CLEANUP_PLAN.md](docs/CLEANUP_PLAN.md) — Fas 0–4 i stort sett klara  
> **Full agent-briefing:** [AGENT_CONTEXT.md](AGENT_CONTEXT.md)

## Nuvarande läge

| Område | Status |
|--------|--------|
| Core pipeline + registry | ✅ Stabil beta (`pipeline.py` refaktorerad) |
| API (FastAPI) | ✅ Fas 4 sign-off |
| NiceGUI dashboard | ✅ Standard-UI (Streamlit avvecklad) |
| Groq + OpenRouter LLM | ✅ Dynamisk pricing via `model_catalog` |
| Edge AI (Fas 5) | 🟡 Grundstruktur (`src/edge/`) |
| LLM model catalog | ✅ CLI + dashboard Test Lab |
| Grok Build skills | ✅ `.grok/skills/` (6 custom skills) |
| Analyzer DX | ✅ `sentimentanalys new-analyzer` + mall |
| Dokumentation | ✅ Konsoliderad (arkiv i `docs/archive/`) |
| CI / tester | ✅ Kör `pytest -q` för live antal |

## Kvar (medveten skuld)

- Ytterligare pipeline-extraktion (LLM-routing → `pipeline_steps.py`) om filen ska under ~500 rader
- Fas 3.1: helt avveckla `requirements*.txt` till förmån för endast `pyproject.toml` optional-deps
- Full production fine-tuning training loop
- Edge AI network + Fas 6 commercialization (se `SYNERGY_ANALYSIS_Fas5.md`)

## Nästa prioriteringar (v0.5)

1. Fine-tuning pipeline + domändata (Fas 2)
2. Produktion: observability, GPU Docker, secrets management
3. Förbättra holistisk LLM-insikt och coaching (inte fler tunna regelanalyzers)
4. Model picker + cost-optimized routing baserat på catalog
