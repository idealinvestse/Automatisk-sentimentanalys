# Städplan — Automatisk-sentimentanalys

**Skapad:** 2026-06-28  
**Syfte:** Prioriterad plan för att minska dokumentationsdrift, arkitekturinkonsistens och legacy-skuld.  
**Kanonical status:** `docs/ROADMAP.md` (engelsk, uppdateras med kod) + denna fil för städarbete.

---

## Fas 0 — Gör nu (låg risk, hög effekt)

| # | Åtgärd | Fil(er) | Status |
|---|--------|---------|--------|
| 0.1 | Rätta analyzer-exemplet (`Analyzer` + `@register_analyzer`, inte `BaseAnalyzer`) | `docs/LLM_AGENT_GUIDE.md` | ✅ |
| 0.2 | Uppdatera stale stubs (llm_judge, webhook) | `docs/ROADMAP.md` | ✅ |
| 0.3 | Synka projektstatus med v0.4.1 | `PROJECT_STATUS.md` | ✅ |
| 0.4 | Ta bort redan levererade INSIGHT-uppgifter | `RECOMMENDED_NEXT_TASKS.md` | ✅ |
| 0.5 | Förtydliga att Fas 6-dashboard är **klar** | `ROADMAP.md` (root) | ✅ |
| 0.6 | Koppla `predictive`-adapter till `RiskAnalyzer` | `src/analysis/predictive.py` | ✅ |
| 0.7 | Fixa `meta`-bugg i llm_judge (`dir()` → lokal variabel) | `src/analysis/llm_judge.py` | ✅ |

---

## Fas 1 — Dokumentkonsolidering ✅ (2026-06-28)

**Mål:** En källa per ämne. Agent ska läsa max 3 filer innan kodändring.

### Behåll (canonical)

| Ämne | Fil |
|------|-----|
| Agent-utveckling | `docs/LLM_AGENT_GUIDE.md` |
| Roadmap & status | `docs/ROADMAP.md` |
| Snabbstart | `README.md` |
| API | `docs/API.md` |
| Säkerhet | `SECURITY.md` |
| Changelog | `CHANGELOG.md` |

### Slå ihop / peka om ✅

| Fil | Status |
|-----|--------|
| `ROADMAP.md` (root) | ✅ Stub → `docs/ROADMAP.md`; historik i `docs/archive/ROADMAP_SV.md` |
| `PROJECT_STATUS.md` | ✅ Uppdaterad |
| `AGENT_CONTEXT.md` | ✅ Pekare → `LLM_AGENT_GUIDE.md` |
| `RECOMMENDED_NEXT_TASKS.md` | ✅ Arkiverad; stub i root |

### Raderat ✅

| Fil | Status |
|-----|--------|
| `plan.md` | ✅ |
| `memory/grok-plans/*.md` | ✅ |
| `docs/LLM_AGENT_GUIDE_OpenClaw.md` | ✅ |
| `docs/AGENTS_OpenClaw.md` | ✅ |

### Arkiverat (`docs/archive/`) ✅

11 filer — se `docs/archive/README.md`.

---

## Fas 2 — Kodstädning ✅ (2026-06-28)

| # | Åtgärd | Detalj |
|---|--------|--------|
| 2.1 | ~~`predictive` duplicering~~ | ✅ Adapter använder `RiskAnalyzer` |
| 2.2 | Pipeline-refaktor | ✅ Delvis: `_run_local_analysis`, `_run_fas4_enrichment`, `_build_report` (~720 r kvar) |
| 2.3 | Streamlit-borttagning | ✅ `app/setup_hub.py` borttagen, `streamlit` ur `requirements-desktop.txt`, launcher-dropdown rensad |
| 2.4 | `alerting.py` docstring | ✅ Beskriver httpx POST + retry + circuit breaker |
| 2.5 | Analyzer-mall | ✅ `src/analysis/templates/new_analyzer_template.py` + `sentimentanalys new-analyzer` |
| 2.6 | Registrera analyzers i profiler | ✅ `configs/analyzer_profiles.yaml` (callcenter: empathy, CES, actionable_coaching, …) |

---

## Fas 3 — Beroenden & CI ✅ (2026-06-28 audit)

| # | Åtgärd |
|---|--------|
| 3.1 | ✅ `requirements*.txt` borttagna; endast `pyproject.toml` optional-deps (DEPS-01) |
| 3.2 | ✅ PIPE-01: Fas-4/LLM i `pipeline_steps.py` |
| 3.3 | ✅ OBS-01: `/metrics` + `docs/PRODUCTION_CHECKLIST.md` |
| 3.4 | ✅ Dokumentera coverage ärligt: `fail_under=65` + omit-lista i `CONTRIBUTING.md` |

---

## Fas 4 — Naming & DX ✅ (2026-06-28, utom 4.1 fasnumrering)

| # | Åtgärd |
|---|--------|
| 4.1 | Enhetlig fasnumrering: använd **Fas N** i svensk docs, **Phase N** endast i äldre engelska summaries |
| 4.2 | ✅ Docs uppdaterade till `samples/audio/sv/` (ingen `samples/call.wav`) |
| 4.3 | ✅ `.gitignore` har `.vs/` och cache är bortstädad |

---

## Verifiering efter varje fas

```bash
pytest -q
ruff check src tests app
# Live test count (ersätter hårdkodade siffror i docs):
pytest --collect-only -q | tail -1
```

---

## Prioriterad ordning om tiden är knapp

1. **Fas 0** — gjort i denna session  
2. **Fas 1** — ✅ dokumentkonsolidering klar (2026-06-28)  
3. **Fas 2.3 Streamlit** — minskar användarförvirring på Windows  
4. **Fas 2.2 Pipeline** — underhållbarhet  
5. **Fas 3** — ny utvecklare onboarding  

---

## Ägarskap

| Roll | Ansvar |
|------|--------|
| Varje PR som ändrar beteende | Uppdatera `docs/ROADMAP.md` stubs-tabell + `CHANGELOG.md` |
| Veckovis (eller vid release) | `PROJECT_STATUS.md` eller ta bort den |
| Agent (Cursor/Windsurf) | Läs **endast** `AGENTS.md` → `docs/LLM_AGENT_GUIDE.md` → `docs/ROADMAP.md` |
