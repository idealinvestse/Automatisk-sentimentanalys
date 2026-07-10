---
artifact_contract: ce-unified-plan/v1
artifact_readiness: completed
product_contract_source: ce-plan-bootstrap
execution: code
title: "Progression v0.5 — nästa djupdykningar"
created: 2026-07-09
completed: 2026-07-09
---

# Progression v0.5 — status

## v0.5 levererat (2026-07-09)

| Spår | Status | Leverabler |
|------|--------|------------|
| **DATA-01** | ✅ | `import_domain_corpus.py`, runbook, baselines refreshed |
| **Docker staging** | ✅ | `docker-compose.staging.yml`, observability smoke script |
| **Model A/B** | ✅ | `POST /analyze_pipeline/compare`, Testlabb panel |
| **INSIGHT-02** | ✅ | `tests/test_insight02_deep_path.py` (9 tester) |
| **CI** | ✅ | mypy job, staging compose config, finetune-smoke |
| **GPU** | ✅ | `verify_gpu_docker.ps1` + checklista uppdaterad |
| **Release** | ✅ | v0.5.0 bump, CHANGELOG/ROADMAP/AGENT_CONTEXT |

**Version:** `0.5.0`

## Kvar post-v0.5

1. **Riktig korpus** — import via `data/import/` när extern data finns (workflow redo)
2. **Intent fine-tune modell** — slå heuristic + 0.05 macro F1 (`compare_intent_backends.py`)
3. **OTLP tracing** — produktionsexporter istället för console
4. **Fas 6** — commercialization

## Session-logg

```
[✅] Pipeline-fix pushad (672cfa2)
[✅] U1 DATA import workflow
[✅] U2 baseline eval smoke
[✅] U3 finetune CI + intent compare
[✅] U4–U6 Docker staging + CI
[✅] U7–U8 Model A/B API + webui
[✅] U9 INSIGHT-02 tester
[✅] U10 GPU verify
[✅] U11 v0.5.0 release docs
```
