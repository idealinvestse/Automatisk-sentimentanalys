# Swedish Sentiment API (v0.4.0)

REST API for Swedish sentiment analysis, ASR transcription, call-center pipelines, and Fas 4 aggregates.

**Interactive docs:** `http://localhost:8000/docs` (uvicorn)  
**OpenAPI JSON:** `/openapi.json`  
**Hardening plan:** [API_REVIEW_HARDENING_PLAN.md](./API_REVIEW_HARDENING_PLAN.md)

---

## Quickstart

```bash
# Dev (no API key)
uvicorn src.api:app --reload --port 8000

# Production-style
export SENTIMENT_API_KEY="your-secret"
export API_MEDIA_ROOT="/var/sentiment/media"
export API_CORS_ORIGINS="https://dashboard.example.com"
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

```bash
curl http://localhost:8000/health
curl -H "X-API-Key: your-secret" http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Det här var jättebra service!"]}'
```

---

## Authentication

| Endpoint | Auth |
|----------|------|
| `GET /health` | None |
| All `POST` routes | Optional `X-API-Key` when `SENTIMENT_API_KEY` is set |

Without `SENTIMENT_API_KEY`, the API accepts requests locally (tests/dev only).

---

## LLM / OpenRouter keys

Prefer header (never log keys):

```http
X-OpenRouter-Key: sk-or-...
```

Body field `llm_api_key` is **disabled by default**. Enable only in controlled environments:

```bash
export API_ALLOW_CLIENT_LLM_KEY=true
```

Server-side `OPENROUTER_API_KEY` (env or `configs/openrouter.key`) is always preferred.

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `SENTIMENT_API_KEY` | Enables `X-API-Key` on mutating routes |
| `API_MEDIA_ROOT` | Restrict `audio_path` / `directory` to this tree |
| `API_CORS_ORIGINS` | Comma-separated CORS allowlist |
| `API_ALLOW_CLIENT_LLM_KEY` | Allow `llm_api_key` in JSON body |
| `API_CACHE_DIR` | Aggregate cache directory (default `.cache/aggregates`) |
| `API_USE_REDIS_CACHE` | `true` to use Redis (`REDIS_URL`) |
| `OPENROUTER_API_KEY` | Mistral/OpenRouter for LLM paths |

---

## Payload limits (Fas 4)

| Limit | Value |
|-------|-------|
| Segments per call | 200 |
| Calls in `segments_list` | 50 |
| Semantic `query` length | 500 chars |

---

## Endpoints

### Health

- `GET /health` — liveness (`{"status": "ok"}`)

### Sentiment

- `POST /analyze` — batch text sentiment (`AnalyzeRequest` → `AnalyzeResponse`)

### Transcription

- `POST /transcribe` — single file ASR
- `POST /batch_transcribe` — parallel ASR (`audio_paths` or `directory` + `glob`)
- `WS /ws/transcription` — real-time log/progress stream during transcription (optional `?api_key=` when auth enabled). Send header `X-Transcription-Job-Id` on POST requests to correlate events. Event types: `log`, `progress`, `status`, `done`.

### Conversation

- `POST /analyze_conversation` — transcribe + per-segment sentiment
- `POST /batch_analyze_conversation` — batch variant

### Full pipeline

- `POST /analyze_pipeline` — full `CallAnalysisPipeline` on pre-transcribed segments  
  Response `results` includes Fas 4 fields: `agent_performance`, `qa` / `compliance_qa`, `agent_assessment`, `customer_metrics`, `alerts`, etc.

### Fas 4 (call center)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/agent_performance/{agent_id}` | Cached agent metrics (`cached` reflects real cache hit) |
| POST | `/search/semantic` | Hybrid semantic search over calls |
| POST | `/insights/hot_topics` | Hot topics / trends |
| POST | `/qa/score` | QA / compliance scoring |
| POST | `/alerts` | Per-call or aggregate alerts |

Shared flags on Fas 4 bodies: `use_mistral_llm`, `deep_analysis`, `llm_model`, `profile`.

### Scan

- `POST /scan_process` — incremental directory scan (`transcribe` or `analyze_conversation`), optional `state_file`

---

## Examples

### Pipeline (segments already transcribed)

```bash
curl -X POST http://localhost:8000/analyze_pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "segments": [{"text": "Hej, tack för att du ringer.", "speaker": "agent"}],
    "profile": "default",
    "use_mistral_llm": false,
    "deep_analysis": false
  }'
```

### Agent performance

```bash
curl -X POST http://localhost:8000/agent_performance/Agent-42 \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "Agent-42",
    "segments_list": [[{"text": "Hej"}]],
    "window": "7d",
    "deep_analysis": false
  }'
```

### Python client

```python
import httpx

client = httpx.Client(base_url="http://localhost:8000", headers={"X-API-Key": "secret"})
r = client.post("/analyze", json={"texts": ["Bra service!"]})
print(r.json()["results"])
```

---

## Errors

| Code | Meaning |
|------|---------|
| 401 | Missing/invalid `X-API-Key` |
| 422 | Validation (Pydantic) or `ConfigurationError` |
| 500 | Analysis/transcription failure (sanitized `detail`) |
| 502 | `LLMError` (OpenRouter/Mistral) |

Responses include `X-Request-ID` for tracing.

---

## Testing

```bash
pytest tests/test_api.py tests/test_api_coverage.py \
  --cov=src/api --cov-fail-under=90
```

Coverage target for `src/api/`: **≥ 90%** (currently ~97% in CI).