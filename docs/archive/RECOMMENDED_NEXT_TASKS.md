# Rekommenderade nästa uppgifter

**Uppdaterad:** 2026-06-28  
**Källor:** användarinput 2026-06-27 + kodgranskning 2026-06-28

## Högsta prioritet

1. **DOC-01** Dokumentstädning enligt [docs/CLEANUP_PLAN.md](docs/CLEANUP_PLAN.md) Fas 1  
   (radera dubbletter, en canonical roadmap, arkivera sign-off-planer)

2. **EXT-01** Developer Experience för nya analyzers  
   - `src/analysis/templates/analyzer_skeleton.py`  
   - CLI: `sentimentanalys new-analyzer <name>`  
   - Uppdatera `docs/LLM_AGENT_GUIDE.md` §5.1 (✅ exempel rättat 2026-06-28)

3. **REF-01** Pipeline-refaktor  
   - Flytta fler steg från `pipeline.py` till `pipeline_steps.py`  
   - Mål: &lt;400 rader i `CallAnalysisPipeline`

## Medel prioritet

4. **INSIGHT-02** Förbättra trajectory + holistisk LLM-insikt (djupare coaching, inte fler tunna regelanalyzers)

5. **LEGACY-01** Ta bort Streamlit-rester (`app/setup_hub.py`, launcher-dropdown, `requirements-desktop.txt`)

## Redan levererat (ta bort från backlog)

| Uppgift | Modul |
|---------|--------|
| EmpathyScore | `src/analysis/empathy_scoring.py` (`empathy`) |
| Customer Effort (CES) | `src/analysis/customer_effort.py` |
| ActionableCoaching | `src/analysis/actionable_coaching.py` |
| LLM Judge | `src/analysis/llm_judge.py` |
| Alerting webhook | `src/alerting.py` `notify_webhook()` |

**Mål:** Insikter som förändrar coaching i svensk kundtjänst — via färre, bättre analyzers + tydligare docs, inte fler parallella markdown-filer.
