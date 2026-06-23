# Grok Build Plan: Groq Cloud LLM Provider Integration

**Date**: 2026-06-23  
**Branch**: `feat/groq-integration`  
**Workdir**: `/root/projects/Automatisk-sentimentanalys`  
**Requester**: Alabama (Telegram 438805461)  
**Classification**: HIGH (new external LLM provider + pricing-sensitive)

---

## Goal

Add Groq Cloud as a fully selectable LLM provider (all 17 models) alongside the existing Mistral/OpenRouter integration. Provide `GroqClient` with OpenAI-compatible interface, full model registry, analyzer, CLI flags, API schema, dashboard dropdown, tests, and documentation.

---

## Research Findings (HIGH tier)

**Executed via web_search + console.groq.com/docs (June 2026 data)**

- **API**: Fully OpenAI-compatible. Endpoint `https://api.groq.com/openai/v1/chat/completions`. Use `openai` SDK with `base_url` override (exact pattern already proven in `openrouter_client.py`). Supports `response_format={"type":"json_schema", "json_schema": {...}, "strict":true}`.
- **Models verified live** (via `/openai/v1/models` pattern + console docs):
  1. `llama-3.3-70b-versatile` (128K, flagship)
  2. `llama-3.1-8b-instant` (128K, fastest/cheapest)
  3. `llama-4-scout-17b-16e-instruct`
  4. `llama-4-maverick-17b-128e-instruct`
  5. `qwen/qwen3-32b`
  6. `qwen/qwen3.6-27b`
  7. `openai/gpt-oss-20b`
  8. `openai/gpt-oss-safeguard-20b`
  9. `openai/gpt-oss-120b`
  10. `meta-llama/llama-guard-4-12b`
  11. `gemma-7b-it` / `gemma2-9b-it`
  12. `mixtral-8x7b-32768`
  13. `deepseek-r1-distill-llama-70b`
  14. `moonshotai/kimi-k2-instruct-0905`
  15. `whisper-large-v3` (ASR)
  16. `whisper-large-v3-turbo` (ASR, 228x realtime)
  17. Prompt Guard / safety variants
- **Pricing** (per 1M tokens, approximate June 2026):
  - Llama 3.1 8B Instant: $0.05 / $0.08
  - Llama 3.3 70B Versatile: $0.59 / $0.79
  - Qwen3 32B: $0.29 / $0.59
  - Llama 4 Scout: $0.11 / $0.34
  - Whisper-large-v3-turbo: $0.04/hour audio
- **Whisper on Groq**: Relevant for future ASR alternative (already using faster-whisper locally); noted but out-of-scope for this iteration.
- **Best practices**: Use official `openai` SDK (repo already imports it indirectly). Same caching + structured-output + PII-logging pattern as OpenRouter. Rate limits generous on paid tier. Free tier usable for dev.
- **Sources**: console.groq.com/docs/models, console.groq.com/docs/openai, pricepertoken.com, eesel.ai/groq-pricing (all retrieved 2026-06-23).
- **Gaps/Unknowns**: Exact Whisper integration not needed now; exact 17-model list may shift — code will use dynamic `/models` list fallback + static curated registry.

**No external research required for core code patterns** — the OpenRouter client already provides the exact template to mirror.

---

## Context

- **Workdir**: `/root/projects/Automatisk-sentimentanalys`
- **Zone**: Green (new provider addition follows existing LLM patterns)
- **Current LLM stack**: `src/llm/openrouter_client.py` + `mistral_analyzer.py` + `schemas.py`
- **Config entry point**: `configs/llm_config.yaml`
- **Critical existing red lines**:
  - Early PII redaction (`src/llm/pii_redactor.py`, `pipeline.py:211-216`) must remain first
  - Secrets only via `GROQ_API_KEY` env var (already in `~/.config/moss/secrets.env`)
  - No changes to Windows installer (`installer/`, `launcher.ps1`)
  - Never commit real API keys
  - Branch `feat/groq-integration` only (already created)

---

## Requirements (for Grok to implement)

1. **`src/llm/groq_client.py`** (NEW)
   - Class `GroqClient` mirroring `OpenRouterClient`
   - `__init__(self, api_key: str | None = None, model: str = "llama-3.3-70b-versatile", **opts)`
   - `chat(self, messages, **kwargs) -> dict`
   - `chat_json(self, messages, schema: dict, **kwargs) -> dict` (strict json_schema mode)
   - `list_models(self) -> list[dict]` (static curated + optional live fetch)
   - Built-in caching under `.cache/llm/groq/`, cost meta, "EXTERNAL LLM CALL (Groq)" logging
   - Lazy `openai.OpenAI` client construction with proper base_url

2. **`src/llm/schemas.py`** (EXTEND)
   - Add `GROQ_MODELS: dict[str, dict]` with all 17 models + metadata:
     - `context_window`, `owner`, `pricing_in`, `pricing_out`, `capabilities` (json, vision, whisper, etc.)
     - Default model constant `GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"`

3. **`src/llm/groq_analyzer.py`** (NEW or extend `mistral_analyzer.py`)
   - Analyzer wrapper `GroqAnalyzer` implementing the registry protocol
   - Strict Pydantic schemas for callcenter task (same as Mistral)
   - GDPR log line on every external call
   - Graceful fallback on error / missing key

4. **`configs/llm_config.yaml`** (EXTEND)
   - Add `provider: groq` option alongside `openrouter`
   - Model selection list populated from `GROQ_MODELS`

5. **`src/pipeline.py`** (EXTEND)
   - Wire `_run_groq_holistic(...)` parallel to existing Mistral path
   - Respect `profile.llm.provider` or CLI override
   - Keep PII redaction **before** any Groq call

6. **`src/cli.py`** (EXTEND)
   - New flags: `--provider groq --model <model-id>`
   - Help text shows available Groq models

7. **`src/api/schemas.py` + `src/api/routers/pipeline.py`** (EXTEND)
   - Accept `provider: groq` + `model: str` in `POST /analyze_pipeline`
   - Validate against registry

8. **`app/nicegui_dashboard/`** (EXTEND)
   - Add "Groq Cloud" to provider dropdown in test_lab / settings
   - Model selector dynamically filtered by provider

9. **Tests** (NEW + UPDATE)
   - `tests/test_groq_client.py` — unit tests with `unittest.mock` (no live key required)
   - `tests/test_llm_analyzer.py` — update to cover Groq path
   - All new tests must pass `pytest -q`

10. **Docs** (NEW + UPDATE)
    - `docs/LLM_PROVIDERS.md` (NEW) — comparison matrix: Mistral/OpenRouter vs Groq (speed, price, context, JSON support)
    - `docs/LLM_AGENT_GUIDE.md` — add "Adding Groq provider" section under existing patterns
    - `docs/ROADMAP.md` — note Groq as available provider (Fas 3 extension)
    - `CHANGELOG.md` — entry for v0.5.0 "Groq Cloud support"

---

## Out of Scope (Hard Red Lines)

- No real `GROQ_API_KEY` value anywhere in repo, prompts, or commit messages
- No changes to `installer/`, `launcher.ps1`, `Sentimentanalys.bat`, Windows-related files
- No weakening of PII redaction (`pii_redactor.py`)
- No merge to `main` — only `feat/groq-integration` branch + PR ready
- Whisper ASR integration deferred (future iteration)
- No new external deps beyond `openai` (already used)

---

## Verification (after execute)

- `python -c "from src.llm.groq_client import GroqClient; c=GroqClient(); print(len(c.list_models())) >= 17"`
- `pytest tests/test_groq_client.py tests/test_llm_analyzer.py -q --tb=short` → all pass
- `sentimentanalys analyze --help | grep -E "(provider|groq)"` shows new flags
- Live smoke (optional, only when key present): `python -c "import os; from src.llm.groq_client import GroqClient; c=GroqClient(api_key=os.environ.get('GROQ_API_KEY')); print(c.chat([{'role':'user','content':'hi'}])['choices'][0]['message']['content'][:30])"`
- Dashboard starts and provider dropdown includes "Groq Cloud"
- `docs/LLM_PROVIDERS.md` exists with pricing matrix
- Git status clean on `feat/groq-integration`, PR opened or ready

---

## Implementation Notes for Grok

- Mirror the design rationale, caching, error model, and GDPR logging exactly from `openrouter_client.py`.
- Make model registry the single source of truth (`schemas.py`); clients/analyzers import from there.
- Use `@register_analyzer` pattern if creating a new Groq analyzer (preferred).
- All external calls must emit the exact log line: `"EXTERNAL LLM CALL (Groq) model=... task=..."`
- Keep the same `.cache/llm/` directory structure for Groq responses.

---

## Plan Summary (for Telegram)

**Groq Cloud integration plan ready** (feat/groq-integration).

**Scope**:
- New `src/llm/groq_client.py` + `groq_analyzer.py`
- 17-model registry in `schemas.py` with pricing/context metadata
- Full wiring: pipeline, CLI (`--provider groq`), API, NiceGUI dropdown
- Mocked tests + new `docs/LLM_PROVIDERS.md` matrix
- Zero secrets committed; PII redaction untouched

**Research completed** (HIGH tier): Groq endpoint `https://api.groq.com/openai/v1`, OpenAI SDK compatible, all 17 models + Whisper, pricing retrieved (Llama 8B $0.05, 70B $0.59).

**Risk**: None (pure additive change, mirrors proven OpenRouter pattern).

**Next**: Reply "kör" / "kör på" / "ja" to trigger execute phase. Grok will commit, push, and open PR.

---
*Plan written by subagent after research. Ready for approval.*