# LLM Providers Comparison

This document provides a comparison matrix of the supported LLM providers in Automatisk-sentimentanalys: **Mistral/OpenRouter** and **Groq Cloud**.

## At a Glance

| Feature | Mistral (via OpenRouter) | Groq Cloud |
|---------|--------------------------|------------|
| **Endpoint** | `https://openrouter.ai/api/v1` | `https://api.groq.com/openai/v1` |
| **API Key** | `OPENROUTER_API_KEY` | `GROQ_API_KEY` |
| **SDK** | `openai>=1.30` (OpenAI-compatible) | `openai>=1.30` (OpenAI-compatible) |
| **Data Centers** | US + varies by model provider | US + Saudi Arabia |
| **EU Data Residency** | Mistral routes via European infra | ❌ None |
| **GDPR-ready** | Yes (European model, privacy-first) | ⚠️ Only with PII redaction or explicit DPA |
| **Default Model** | `mistralai/mistral-medium-3.5` | `llama-3.3-70b-versatile` |
| **Strict JSON Schema** | ✅ (all Mistral models) | ⚠️ Only `gpt-oss-20b` / `gpt-oss-120b` |
| **Fastest Model** | ~80 tps | ~840 tps (Llama 8B) |
| **Cheapest Model** | $1.50 input / $7.50 output (per 1M) | $0.05 input / $0.08 output (per 1M) |
| **Max Context** | 256K | 131K |
| **Streaming** | ✅ | ✅ (mutually exclusive with structured outputs) |
| **Free Tier** | No (credits available) | ✅ (rate-limited) |
| **Provider Flag** | `--provider openrouter` | `--provider groq` |
| **GDPR Gate** | N/A (European by default) | `--groq-eu-residency` or PII redaction |

## Model Comparison

### Mistral / OpenRouter

| Model | Context | Cost (In/Out per 1M) | Best For |
|-------|---------|----------------------|----------|
| `mistralai/mistral-medium-3.5` | 256K | $1.50 / $7.50 | Default, balanced quality |
| `mistralai/mistral-large-3` | 256K | $2.00 / $6.00 | Complex/high-value calls |
| `mistralai/mistral-large-2512` | 256K | $2.00 / $6.00 | Latest Mistral flagship |

### Groq Cloud

| Model | Context | Cost (In/Out per 1M) | Best For |
|-------|---------|----------------------|----------|
| `llama-3.1-8b-instant` | 131K | $0.05 / $0.08 | Fastest, cheapest (~840 tps) |
| `llama-3.3-70b-versatile` | 131K | $0.59 / $0.79 | Best quality/price (~275 tps) |
| `openai/gpt-oss-20b` | 131K | $0.075 / $0.30 | Strict JSON schema (~1000 tps) |
| `openai/gpt-oss-120b` | 131K | $0.15 / $0.60 | Large strict schema (~500 tps) |
| `meta-llama/llama-4-scout-17b-16e-instruct` | 131K | $0.11 / $0.34 | Vision + tools (MoE) |
| `qwen/qwen3-32b` | 131K | $0.29 / $0.59 | Strong reasoning + tools |
| `qwen/qwen3.6-27b` | 131K | $0.60 / $3.00 | Vision + text |
| `groq/compound` | 131K | FREE (preview) | Agentic routing |
| `whisper-large-v3-turbo` | N/A | $0.04/hr audio | ASR (228× realtime) |
| `whisper-large-v3` | N/A | $0.111/hr audio | ASR (217× realtime) |

Full model registry: see `src/llm/schemas.py` → `GROQ_MODELS` (17 models total).

## GDPR / Data Residency

### Mistral / OpenRouter

- **Mistral models are European** (France-based). When routed via OpenRouter with `provider="Mistral"`, data processing happens within European infrastructure.
- **GDPR-compliant by default** for EU customer data.
- PII redaction is still strongly recommended per `SECURITY.md`.

### Groq Cloud

- ⚠️ **Groq data centers are in the United States and Saudi Arabia.**
- **No confirmed EU data residency.**
- This is the **single biggest blocker** for GDPR-sensitive workloads.

**Mitigation required (enforced in pipeline):**

1. **`groq_eu_residency` config flag** (`configs/llm_config.yaml` → `groq.groq_eu_residency`) — set to `true` only if you have a DPA with Groq or accept non-EU processing.
2. **`anonymize_before_llm`** — PII redaction runs BEFORE any Groq call. Set in profile config.
3. **Pipeline enforcement:** If `groq_eu_residency=False` AND no PII redaction detected, the pipeline **rejects the Groq call** and falls back to local analysis.

## Recommended Provider Per Use Case

| Use Case | Recommended Provider | Reason |
|----------|---------------------|--------|
| Production call center (EU data) | Mistral / OpenRouter | European hosting, GDPR-compliant |
| Dev / prototyping (no PII) | Groq (Llama 8B) | Free tier, ~840 tps |
| Structured JSON analysis | Mistral (strict schema on all models) | Reliable strict mode |
| Structured JSON analysis (Groq) | Groq (`gpt-oss-20b`) | Only Groq model with strict schema |
| High-throughput batch (no PII) | Groq (Llama 8B) | $0.05 input, fastest inference |
| Complex reasoning + Swedish | Mistral Medium 3.5 | Best Swedish performance |
| Vision (image analysis) | Groq (Qwen3.6 27B) | Groq's only vision model |
| ASR (cloud alternative to local whisper) | Groq (Whisper turbo) | 228× realtime, $0.04/hr |

## CLI Usage Examples

```bash
# Mistral / OpenRouter (default, EU/GDPR)
python -m src.cli analyze-call call.wav --use-mistral-llm --provider openrouter

# Groq (fast, cheap, but US/Saudi hosting)
python -m src.cli analyze-call call.wav --use-mistral-llm --provider groq \
  --groq-eu-residency --llm-model llama-3.1-8b-instant

# Groq with PII redaction (GDPR mitigation)
python -m src.cli analyze-call call.wav --use-mistral-llm --provider groq \
  --profile callcenter  # requires anonymize_before_llm: true in profile
```

## Configuration

See `configs/llm_config.yaml` for provider-specific settings:

```yaml
llm:
  provider: "openrouter"  # or "groq"

# Groq specific
groq:
  default_model: "llama-3.3-70b-versatile"
  fast_model: "llama-3.1-8b-instant"
  strict_schema_model: "openai/gpt-oss-20b"
  fallback_chain:
    - "llama-3.1-8b-instant"
    - "llama-3.3-70b-versatile"
    - "openai/gpt-oss-20b"
  groq_eu_residency: false
  enforce_anonymize_before_llm: true
```

## Fallback Chain

When using Groq, the pipeline implements a configurable fallback chain:

1. `llama-3.1-8b-instant` → cheapest/fastest
2. `llama-3.3-70b-versatile` → better quality
3. `openai/gpt-oss-20b` → strict JSON schema

If all models fail, the system falls back to **local analysis only** (heuristics + ML models, no external API call).

## Limitations

- **Streaming + structured outputs are mutually exclusive on Groq.** Choose one pattern per request.
- **Free tier has TPM (tokens per minute) rate limits.** Production use requires Dev or Paid tier.
- **Model deprecation risk:** Groq's model catalog changes faster than OpenRouter. The code's static registry should be updated periodically.

---

*Updated 2026-06-23 as part of Groq Cloud integration.*