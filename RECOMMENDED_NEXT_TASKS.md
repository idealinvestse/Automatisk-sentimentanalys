# RECOMMENDED NEXT TASKS — Automatisk-sentimentanalys

**Generated:** 2026-06-27 15:10 CEST by github-repo-deep-dive skill (after github-project-status)  
**Based on:** Fresh PROJECT_STATUS.md + AGENT_CONTEXT.md + deep code analysis of recent model catalog work

## Understanding & Rationale
The project has just built a solid foundation for **LLM Model Management** (model_catalog.py, CLI scan command, Dashboard refresh button, dynamic pricing in openrouter_client). 
This is a high-leverage area because:
- It directly addresses cost control and model choice for expensive LLM calls in callcenter analysis.
- It builds on the previous "valbar mapp" feature.
- It is isolated and well-documented, making extension low-risk.

Current state is strong on *scanning & saving*, but weak on *using* the data. The next tasks should focus on making the catalog deliver immediate value (model selection, cost visibility, integration) rather than starting completely new big features.

## Prioritized Task List

### Quick Wins (can be done in 1 session, high visible impact)

**TASK-01: Clean up duplicate code & add .gitignore for catalog**
- **Why now**: Duplicate `fetch_and_save_models_catalog` exists in both model_catalog.py and openrouter_client.py. .gitignore missing entry for the generated JSON (can grow large).
- **Description**: Remove the duplicate method from openrouter_client.py, make it import and re-export from model_catalog. Add `data/openrouter_models_catalog.json` to .gitignore. Update any internal references.
- **Primary files**: `src/llm/openrouter_client.py`, `.gitignore`, `src/llm/model_catalog.py`
- **Effort**: Small
- **Impact**: Cleaner codebase, prevents accidental large file commits.
- **Dependencies**: None
- **Success criteria**: `git status` clean, no duplicate code, catalog file ignored.

**TASK-02: Wire storage path into model catalog save location**
- **Why now**: The previous configurable MODELS_DIR feature exists in launcher/dashboard, but the new catalog always saves to hardcoded `data/`. This breaks the "valbar mapp" UX.
- **Description**: Make `fetch_openrouter_models_catalog` accept optional output_path from config / UserConfig (like ASR assets). Update CLI and Dashboard button to respect the chosen models directory.
- **Primary files**: `src/llm/model_catalog.py`, launcher settings, `app/nicegui_dashboard/components/test_lab.py`, `src/install/config_schema.py`
- **Effort**: Small-Medium
- **Impact**: Consistent UX for all large file downloads (ASR + LLM models).
- **Dependencies**: TASK-01 (clean code)
- **Success criteria**: Changing storage path in dashboard also affects where catalog is saved.

### Strategic / Foundational (builds long-term capability)

**TASK-03: Expose model catalog in llm_config.yaml + profiles**
- **Why now**: llm_config.yaml still has hardcoded `default_model`. The catalog is live but not connected to configuration.
- **Description**: Add `openrouter.model_catalog_path` and `preferred_models` section in llm_config.yaml. Load catalog at startup in OpenRouterClient or a new ModelRegistry. Allow profile to specify "cheapest_strict_schema" etc.
- **Primary files**: `configs/llm_config.yaml`, `src/llm/openrouter_client.py`, `src/profiles.py`, new `src/llm/model_registry.py` (optional)
- **Effort**: Medium
- **Impact**: Foundation for smart model routing and cost budgets.
- **Dependencies**: TASK-01, TASK-02
- **Success criteria**: Changing default_model in config uses catalog data; cost estimates use live prices.

**TASK-04: Improve Dashboard Model Catalog UI (search, filter, "use this model")**
- **Why now**: Current test_lab button is basic. Users need to actually *choose* models from the 200+ list.
- **Description**: Enhance the card with ui.table or ui.select + search, columns for price/context/description. Add button "Använd denna modell för nästa analys" that updates session state or llm_config.
- **Primary files**: `app/nicegui_dashboard/components/test_lab.py`, `app/nicegui_dashboard/services/`
- **Effort**: Medium
- **Impact**: Makes the new feature actually usable in daily workflow.
- **Dependencies**: TASK-03 (config integration)
- **Success criteria**: User can search "mistral" and select a model that then appears in analyze-call --llm-model suggestions.

### New Capability (higher effort, high strategic value)

**TASK-05: Auto cost-optimized model selection for callcenter tasks**
- **Why now**: With live pricing + strict json_schema requirement for CallLLMOutput, we can automatically pick the cheapest model that supports the needed capabilities.
- **Description**: In OpenRouterClient or a new router, before structured_chat, query catalog for models that support json_schema_strict + have acceptable price. Add `auto_cheapest=True` flag.
- **Primary files**: `src/llm/openrouter_client.py`, `src/llm/model_catalog.py`, `src/llm/schemas.py`
- **Effort**: Medium-Large
- **Impact**: Significant cost savings on high-volume callcenter use without quality loss.
- **Dependencies**: TASK-03, TASK-04
- **Success criteria**: When --use-mistral-llm is active, the meta shows the actually used (possibly cheaper) model and lower cost than before.

## How to use this file
1. Pick the top task that matches your current priority (quick win or strategic).
2. Read the relevant sections in `AGENT_CONTEXT.md` + `PROJECT_STATUS.md`.
3. Implement (an agent can start directly from the Description).
4. After done: re-run `github-project-status skill` then optionally this deep-dive skill again.

**Next after these?** Model picker component, integration with fine-tuning model selection, Edge AI node economics using catalog prices.