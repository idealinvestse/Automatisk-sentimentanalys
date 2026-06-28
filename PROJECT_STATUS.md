# Projektstatus - Automatisk Sentimentanalys

**Senast uppdaterad:** 2026-06-28  
**Version:** 0.4.1 → v0.5 (implementering pågår)

> **Canonical roadmap:** [docs/ROADMAP.md](docs/ROADMAP.md)  
> **Analyzer-strategi:** [docs/ANALYZER_STRATEGY.md](docs/ANALYZER_STRATEGY.md)  
> **Städplan:** [docs/CLEANUP_PLAN.md](docs/CLEANUP_PLAN.md)  
> **Produktion:** [docs/PRODUCTION_CHECKLIST.md](docs/PRODUCTION_CHECKLIST.md)  
> **Full agent-briefing:** [AGENT_CONTEXT.md](AGENT_CONTEXT.md)

## Nuvarande läge

| Område | Status |
|--------|--------|
| Core pipeline + registry | ✅ PIPE-01; deep-path skip (INSIGHT-02) |
| Beroenden | ✅ `pyproject.toml` only (DEPS-01) |
| API (FastAPI) | ✅ Fas 4 + `/metrics`; prod-guards (v0.5) |
| NiceGUI dashboard | ✅ Standard-UI |
| Groq + OpenRouter LLM | ✅ + model routing (v0.5) |
| Edge AI | 🟡 MVP: `sentimentanalys edge-analyze` |
| Observability | 🟡 JSON logs + pipeline/LLM metrics (PROD-01) |
| Fine-tuning CI | 🟡 Smoke + baseline eval (DATA-01) |
| Dokumentation | ✅ v0.5 sync + root archive |

## Nästa prioriteringar (v0.5)

1. PROD-01: tracing, full metrics dashboards
2. DATA-01: nightly fine-tune eval on expanded corpus
3. Real domain data import (GDPR-safe workflow)
4. Dashboard polish (Executive Insights, model A/B)

## Kvar (medveten skuld)

- OpenTelemetry production deployment guides
- Real annotated call corpus (1000+ production calls)
- Fas 6 commercialization (post v0.5)
