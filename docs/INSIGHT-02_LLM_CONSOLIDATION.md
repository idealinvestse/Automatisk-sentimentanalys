# INSIGHT-02 LLM Consolidation

## Mål
- Prioritera deep-path LLM-judge framför heuristiska analyzers
- Minska beroende av lexicon och rule-based
- Behåll hybrid för speed + fallback
- Öka accuracy på svenska call-center data

## Ändringar
- Uppdatera registry.py att använda LLM som default för sentiment/emotion/intent
- Lägg till cost/latency guard i routing
- Nightly eval inkluderar LLM-consolidation metrics

Status: Påbörjat 2026-07-10