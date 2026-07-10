# Projektstatus - Automatisk Sentimentanalys

**Senast uppdaterad:** 2026-07-10  
**Version:** 0.5.0

> **Canonical roadmap:** [docs/ROADMAP.md](docs/ROADMAP.md)  
> **Analyzer-strategi:** [docs/ANALYZER_STRATEGY.md](docs/ANALYZER_STRATEGY.md)  
> **Städplan:** [docs/CLEANUP_PLAN.md](docs/CLEANUP_PLAN.md)  
> **Produktion:** [docs/PRODUCTION_CHECKLIST.md](docs/PRODUCTION_CHECKLIST.md)  
> **Full agent-briefing:** [AGENT_CONTEXT.md](AGENT_CONTEXT.md)

## Nuvarande läge

| Område | Status |
|--------|--------|
| Core pipeline + registry | ✅ PIPE-01; deep-path skip (INSIGHT-02) + tester |
| Beroenden | ✅ `pyproject.toml` only (DEPS-01) |
| API (FastAPI) | ✅ Fas 4 + `/metrics`; prod-guards; model A/B compare |
| NiceGUI dashboard | ✅ Legacy (deprecated, se webui/) |
| Next.js webui (webui/) | ✅ Primär dashboard — Fas 1-4 + model compare i Testlabb |
| Groq + OpenRouter LLM | ✅ + model routing + side-by-side compare (v0.5) |
| Edge AI | 🟡 MVP: `sentimentanalys edge-analyze` + REST `/edge/*` |
| Observability | ✅ JSON logs + pipeline/LLM metrics + Docker staging stack |
| DATA-01 | ✅ Import workflow + baseline eval smoke + CI gates |
| Fine-tuning CI | ✅ Smoke + baseline eval + intent backend comparison |
| Docker staging | ✅ `docker-compose.staging.yml` (api+webui+redis+prometheus) |
| Dokumentation | ✅ v0.5 release sync |

## v0.5 leveranser (2026-07-09)

1. **DATA-01** — import runbook, `import_domain_corpus.py`, refreshed baselines
2. **Docker staging** — production-like compose + observability smoke script
3. **Model A/B** — `POST /analyze_pipeline/compare` + Testlabb panel
4. **INSIGHT-02** — verifierade tester för `skip_llm_superseded`
5. **CI** — mypy job + staging compose config validation
6. **GPU** — verifieringsscript + checklista (se PRODUCTION_CHECKLIST §3)

## Nästa prioriteringar (post v0.5)

1. Riktig anonymiserad korpus (1000+ samtal) via DATA-01 import-slot
2. Intent fine-tune modell som slår heuristic + 0.05 macro F1
3. Svensk ASR-pack + L7–L9 release-verifiering före pilot
4. Fas 6 commercialization

## Kvar (medveten skuld)

- Real annotated call corpus (väntar på extern data)
- Svensk audio under `samples/audio/sv/` (kataloger finns, inga wav)
- WS event hub fortfarande process-lokal (tickets är Redis-redo via `TicketStore`)
- Windows keyring för launcher secrets
