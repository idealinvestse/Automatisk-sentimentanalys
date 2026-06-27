# Project Status — Automatisk-sentimentanalys

**Last Updated:** 2026-06-27 15:00 CEST via github-project-status skill + model catalog feature

## Recent Activity
- Configurable storage path (disk/mapp) for LLM models & downloads selectable in Launcher + Dashboard (previous TASK)
- **NEW: OpenRouter model catalog scanner** (`src/llm/model_catalog.py`)
  - Scans https://openrouter.ai/api/v1/models
  - Saves id, name, short description, context_length, full pricing (per token + per million USD)
  - JSON catalog with timestamp (data/openrouter_models_catalog.json)
  - Usable from Python, future CLI `scan-openrouter-models` and Dashboard refresh button
- Updated openrouter_client.py with helper methods

## Feature Status
### Implemented
- ... (previous features)
- Dynamic LLM model discovery for OpenRouter (all ~200+ models with live pricing + info)
- Integration ready for dashboard model picker and cost estimation

**Next immediate:** Add Typer command in src/cli.py and NiceGUI button in dashboard settings.

Systemet är nu redo för smart model-val baserat på kostnad/prestanda för call center workloads!