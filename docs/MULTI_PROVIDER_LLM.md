# Multi-provider LLM router

## Providers

| Provider   | Base URL | Key (env / file) |
|------------|----------|------------------|
| openrouter | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` / `configs/openrouter.key` |
| mistral    | `https://api.mistral.ai/v1` | `MISTRAL_API_KEY` / `configs/mistral.key` |
| nvidia     | `https://integrate.api.nvidia.com/v1` | `NVIDIA_API_KEY` / `configs/nvidia.key` |
| cerebras   | `https://api.cerebras.ai/v1` | `CEREBRAS_API_KEY` / `configs/cerebras.key` |
| groq       | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` (dev-only; GDPR) |

Keys are **gitignored** (`*.key`, `.env`). Never commit them.

Config SSOT: [`configs/llm_providers.yaml`](../configs/llm_providers.yaml)

## Catalog scan

```bash
# all configured providers → data/model_catalogs/<provider>.json + index.json
python -m src.llm.model_catalog

# single provider
python -m src.llm.model_catalog nvidia
```

Each catalog entry has: `id`, `name`, `description`, `context_length`, `pricing`, `is_free`, `provider`.  
If a provider returns 403/401 on `/models` (e.g. Cerebras), the scanner **seeds** from curated free/sv lists in yaml.

## Router profiles

### `free_sequential` (default via `LLM_ROUTER_PROFILE`)
- Only free / free-tier models
- **One provider at a time** with RPM window + cooldown on 429
- Failover to next provider if call fails
- Order (default): nvidia → cerebras → mistral → openrouter

### `sv_optimal`
- Best Swedish-capable curated models per provider
- Sequential by default; `map_parallel()` for independent tasks

## API / pipeline

```json
POST /analyze_pipeline
{
  "segments": [...],
  "use_mistral_llm": true,
  "deep_analysis": true,
  "provider": "free_sequential"
}
```

Allowed `provider` values:  
`openrouter | groq | mistral | nvidia | cerebras | auto | free_sequential | sv_optimal | router`

## Code map

- `src/llm/provider_secrets.py` — key resolution
- `src/llm/openai_compat_client.py` — OpenAI-compatible client
- `src/llm/model_catalog.py` — multi-provider scanner
- `src/llm/multi_provider_router.py` — routing + rate limits
- `src/llm/router_client.py` — analyzer-compatible adapter
- `src/pipeline_steps.py` — wires providers into holistic LLM path

## Analysis perspectives (paid, cost-aware)

Selectable profiles for different analysis goals. Each profile has
`cost_priority` / `quality_priority` and a max blended $/M token budget.
The advisor ranks **paid** catalog models and returns a simple menu.

```bash
python -m src.llm.paid_model_advisor
# → data/model_catalogs/analysis_profiles.json
```

```http
GET /llm/analysis-profiles
GET /llm/analysis-profiles/{id}
```

Example pipeline body:

```json
{
  "segments": [...],
  "use_mistral_llm": true,
  "analysis_perspective": "coaching_qa"
}
```

Backend auto-fills `provider` + `llm_model` from the recommendation when
`llm_model` is omitted. UI: Testlabb → “Analysperspektiv (paid modeller)”.

Perspectives include: `cost_saver`, `batch_throughput`, `sentiment_refine`,
`intent_routing`, `root_cause`, `coaching_qa`, `compliance_risk`,
`summary_actions`, `swedish_quality`, `holistic_deep`, `balanced_ops`,
`premium_reasoning`.
