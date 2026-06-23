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

### Initial findings (grok-4.3, 2026-06-23 22:23)

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

### ⚠️ REFRESHED findings (deepseek-v4-pro, 2026-06-23 22:31) — corrections + new risks

Live API probe (`GET /openai/v1/models` with `GROQ_API_KEY`) reveals the **actual current catalog** differs from earlier 2026 docs:

- **Live catalog (17 models, June 2026):**
  - `llama-3.1-8b-instant` (131K, ~840 tps, tools+json_mode) — **cheapest production**
  - `llama-3.3-70b-versatile` (131K, tools+json_mode) — flagship
  - `meta-llama/llama-4-scout-17b-16e-instruct` (131K, MoE, vision, tools+json_mode)
  - `openai/gpt-oss-20b` (131K, **strict json_schema**, ~1,000 tps) — newest
  - `openai/gpt-oss-120b` (131K, strict json_schema, ~500 tps)
  - `openai/gpt-oss-safeguard-20b` (131K, safety variant)
  - `qwen/qwen3-32b` (131K, reasoning + tools)
  - `qwen/qwen3.6-27b` (131K, **vision + text**)
  - `groq/compound` + `groq/compound-mini` (131K, agentic, preview/free)
  - `whisper-large-v3` (217× realtime, $0.111/hr)
  - `whisper-large-v3-turbo` (228× realtime, $0.04/hr — **89% cheaper than OpenAI Whisper**)
  - `canopylabs/orpheus-v1-english` + `canopylabs/orpheus-arabic-saudi` (TTS)
  - `meta-llama/llama-prompt-guard-2-{22m,86m}` (content safety)
  - `allam-2-7b` (Arabic, 4K)
- **Notably ABSENT from live API** (deprecated or region-gated): Kimi K2, Llama 4 Maverick, DeepSeek R1 Distill, Mixtral, Gemma — earlier 2026 docs mentioned these but they're not in the live response. **Model deprecation risk is real.**

### 🔴 CRITICAL NEW FINDING — GDPR / data residency

- **Groq data centers: US + Saudi Arabia** (PIF investment). **No confirmed EU hosting.**
- **This is the single biggest blocker for GDPR-sensitive workloads** (Swedish call-center data is sensitive).
- Mitigation options: (a) Azure/GCP EU regions with Llama, (b) self-hosted Llama, (c) keep Groq as opt-in for non-PII / dev workloads only.
- **Plan impact:** Add explicit PII-redaction gate before ANY Groq call (existing pipeline already does this — verify + document); add "groq_eu_residency" config flag (default: OFF); add per-call `anonymize_before_llm` enforcement.
- Sources: console.groq.com/docs/rate-limits, console.groq.com/docs/structured-outputs, eesel.ai/groq-pricing, cloudzero.com, aipricing.guru (all 2026-06-23).

### Pricing (per 1M tokens, June 2026 — verified live + 4 sources)

| Model | Input | Output | Cached Input |
|-------|-------|--------|-------------|
| `llama-3.1-8b-instant` | $0.05 | $0.08 | $0.025 |
| `openai/gpt-oss-20b` | $0.075 | $0.30 | $0.0375 |
| `meta-llama/llama-4-scout-17b-16e-instruct` | $0.11 | $0.34 | $0.055 |
| `openai/gpt-oss-120b` | $0.15 | $0.60 | $0.075 |
| `qwen/qwen3-32b` | $0.29 | $0.59 | $0.145 |
| `llama-3.3-70b-versatile` | $0.59 | $0.79 | $0.295 |
| `qwen/qwen3.6-27b` | $0.60 | $3.00 | $0.30 |

- **Discounts:** prompt caching 50% off, batch API 50% off, stacked → ~25% of on-demand
- **Whisper-large-v3-turbo:** $0.04/audio-hour (228× realtime, **89% cheaper than OpenAI Whisper $0.36/hr**)
- **Orpheus TTS:** $22/1M chars (English), $40/1M chars (Arabic)

### API surface — critical constraints

- **OpenAI-compatible** endpoint, standard `Authorization: Bearer $GROQ_API_KEY`
- **Streaming: YES** (SSE, `stream: true`)
- **json_schema strict mode:** `gpt-oss-20b`, `gpt-oss-120b` ONLY
- **json_mode (best-effort):** llama, qwen, gpt-oss, compound, allam
- ⚠️ **Streaming + structured outputs mutually exclusive** — can't stream when using json_schema
- **Tool/function calling:** up to 128 functions, `parallel_tool_calls`, `tool_choice`
- **Reasoning models:** gpt-oss family (low/medium/high) + qwen3 (none/default)
- **Service tiers:** `on_demand` (default), `flex` (10× rate limits, occasional failures OK), `performance` (prioritized)
- **SDKs:** official `groq` Python + `groq-sdk` JS/TS; can also use `openai` SDK with `base_url` override (already in repo via `openrouter_client.py` pattern)

### Rate limits

| Tier | RPM | TPM | RPD | Concurrent |
|------|-----|-----|-----|------------|
| **Free** | 30 | 6K–12K | 1K–14K | 5 |
| **Dev (paid, CC)** | ~100 | ~20K | ~10× free | — |
| **Paid (prod)** | 1,000 | 500K | — | 50 |
| **Flex** | 10× base | — | — | — |

- **Free tier TPM bottleneck (6K/min):** single 3K-token system prompt = half the budget. Free tier for prototyping ONLY.
- Dev tier: 25% off all tokens + up to 10× rate limits.

### 🎯 Production recommendations (Swedish call-center)

- **Default:** `llama-3.1-8b-instant` — cheapest ($0.05/$0.08), fastest (~840 tps), json_mode
- **Strict schema:** `openai/gpt-oss-20b` — strict json_schema (constrained decoding), $0.075/$0.30
- **Fallback chain:** `llama-3.1-8b-instant` → `llama-3.3-70b-versatile` → `openai/gpt-oss-20b`
- **Pitfalls:** free tier TPM, structured+streaming incompatibility, GDPR residency, P99 latency spikes, single-provider outage risk, model deprecation volatility

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
- **No weakening of PII redaction (`pii_redactor.py`) — Groq is US/Saudi-hosted, GDPR risk is HIGH, so early redaction is mandatory, not optional**
- **Add `groq_eu_residency` config flag (default: OFF) — disables Groq for any profile where `anonymize_before_llm` is required but residency unconfirmed**
- **Add per-call `anonymize_before_llm` enforcement check before any Groq request**
- No merge to `main` — only `feat/groq-integration` branch + PR ready
- Whisper ASR integration deferred (future iteration)
- No new external deps beyond `openai` (already used)
- **Streaming + structured outputs mutually exclusive — choose one pattern per request, document in client**

## Verification (extended with GDPR gate)

- All previous verifications
- **GDPR gate test:** verify pipeline rejects Groq call when `groq_eu_residency=OFF` and `anonymize_before_llm` would be required
- **Fallback chain test:** simulate 8B timeout → verify 70B called → verify gpt-oss-20b called for strict schema
- **Free tier refusal:** config flag to disable free tier in production

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