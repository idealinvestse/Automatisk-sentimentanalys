# Swedish Sentiment API (v0.4.1)

REST API for Swedish sentiment analysis, ASR transcription, call-center pipelines, and Fas 4 aggregates.

**Interactive docs:** `http://localhost:8000/docs` (uvicorn)  
**OpenAPI JSON:** `/openapi.json`  
**Hardening plan:** [API_REVIEW_HARDENING_PLAN.md](./archive/API_REVIEW_HARDENING_PLAN.md)

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
| `GET /health`, `GET /ready` | None |
| `GET /status/*`, `GET /metrics`, `GET /alerting/status` | Optional `X-API-Key` when `SENTIMENT_API_KEY` is set |
| `GET /ws/transcription/ticket` | Optional `X-API-Key` when `SENTIMENT_API_KEY` is set |
| `WS /ws/transcription` | `X-API-Key` **or** `?token=` from `/ws/transcription/ticket` when auth is enabled |
| All other `GET`/`PUT`/`POST`/`DELETE` routes (`/analyze`, `/calls`, `/llm/*`, `/edge/*`, …) | Optional `X-API-Key` when `SENTIMENT_API_KEY` is set |

Browsers cannot set custom headers on the WebSocket handshake. Issue a ticket with `GET /ws/transcription/ticket` (header auth) and connect as `WS /ws/transcription?token=<ticket>`. Tickets expire after 5 minutes. Non-browser clients may send `X-API-Key` on the handshake instead.

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
| `API_RATE_LIMIT_RPM` | Requests/minute per client IP (`0` = disabled, default) |
| `API_TRUSTED_PROXY` | `true` to honor `X-Forwarded-For` for rate limiting (only behind a trusted reverse proxy) |
| `API_STATE_DIR` | Directory allowed for `/scan_process` `state_file` (defaults to `API_CACHE_DIR`) |
| `OPENROUTER_API_KEY` | Mistral/OpenRouter for LLM paths |
| `DEEPGRAM_API_KEY` / `CLOUD_STT_API_KEY` | Deepgram cloud STT (opt-in only; see [SECURITY.md](../SECURITY.md)) |
| `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` | pyannote diarization models (when `diarize=true`) |

### Cloud STT (opt-in)

Default ASR provider is **local**. Cloud sends raw audio to Deepgram only when:

- Request or config sets `provider: "cloud"`, **and**
- `DEEPGRAM_API_KEY` (or `CLOUD_STT_API_KEY`) is set.

`cloud_fallback_local` defaults to `false` (no silent fallback). Egress is logged via `asr_cloud_egress_total` without audio/transcript content. See [SECURITY.md](../SECURITY.md).

Transcription endpoints accept `provider`, `cloud_fallback_local`, `diarize`, and `num_speakers` via `AsrParamsMixin`.

---

## Payload limits (Fas 4)

| Limit | Value |
|-------|-------|
| Texts in `/analyze` | 1000 |
| Segments per call | 200 |
| Calls in `segments_list` | 50 |
| Semantic `query` length | 500 chars |

---

## Endpoints

### Health

- `GET /health` — liveness (`{"status": "ok"}`)
- `GET /ready` — readiness. Returns **200** `{status: ready, checks: …}` or **503** `{status: not_ready, checks: …}` when production-critical deps fail (`SENTIMENT_API_KEY` / `API_MEDIA_ROOT` in production mode, Redis ping when `API_USE_REDIS_CACHE=true`).

### Sentiment

- `POST /analyze` — batch text sentiment (`AnalyzeRequest` → `AnalyzeResponse`)

### Transcription

- `POST /upload` — upload audio to `API_MEDIA_ROOT` (size limits + retention)
- `POST /transcribe` — single file ASR
- `POST /batch_transcribe` — parallel ASR (`audio_paths` or `directory` + `glob`)
- `GET /transcription/jobs` — list background transcription jobs
- `GET /transcription/jobs/{job_id}` — job status
- `POST /transcription/jobs/{job_id}/cancel` — cancel a running job
- `GET /ws/transcription/ticket` — short-lived ticket for WebSocket auth (when `SENTIMENT_API_KEY` is set)
- `WS /ws/transcription` — real-time log/progress stream. Authenticate with `X-API-Key` **or** `?token=` from the ticket endpoint. Missing/invalid/expired credentials close the socket with **1008**. Send header `X-Transcription-Job-Id` on POST requests to correlate events. Event types: `log`, `progress`, `status`, `done`. Client may send `{"type":"ping"}` or `{"type":"subscribe","job_id":"…"}`.

### Conversation

- `POST /analyze_conversation` — transcribe + per-segment sentiment (light path, default)
  - `use_full_pipeline: true` — full `CallAnalysisPipeline` (PII, QA, agent metrics); response includes optional `pipeline_results`
  - `sentiment_profile` — default `callcenter` (light path only)
- `POST /batch_analyze_conversation` — batch variant

### Full pipeline

- `POST /analyze_pipeline` — full `CallAnalysisPipeline` on pre-transcribed segments  
  Response `results` includes Fas 4 fields: `agent_performance`, `qa` / `compliance_qa`, `agent_assessment`, `customer_metrics`, `alerts`, etc.

- `POST /analyze_pipeline/partial` — incremental local analysis on growing segment lists (`previous_results`, `reconcile`). Same `PipelineResponse` shape as `/analyze_pipeline`.

- `POST /analyze_pipeline/compare` — run the same segments through up to 3 LLM models (A/B comparison)  
  Request: `segments`, `models` (max 3 slugs), optional `cost_budget_usd`.  
  Response: per-model `results` with QA score, sentiment, cost, latency, and full `PipelineResponse`.

### Calls (server-side history)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/calls` | List saved analyzed calls (newest first, `limit` 1–500) |
| GET | `/calls/{id}` | Fetch one saved call (**404** if missing) |
| POST | `/calls` | Create or update (`id` in body) |
| PUT | `/calls/{id}` | Create or update; path `id` must match `body.id` (**422** on mismatch) |
| DELETE | `/calls/{id}` | Delete (**404** if missing) |

### LLM profiles

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/llm/analysis-profiles` | Selectable analysis perspectives + paid-model picks (`cached`, `menu`, `providers_configured`) |
| GET | `/llm/analysis-profiles/{id}` | Detail + ranked models for one perspective (**404** `unknown_perspective`) |
| GET | `/llm/providers` | Which LLM providers have a key configured (booleans only — never secret values) |

### Fas 4 (call center)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/agent_performance/{agent_id}` | Cached agent metrics (`cached` reflects real cache hit) |
| POST | `/search/semantic` | Hybrid semantic search over calls |
| POST | `/insights/hot_topics` | Hot topics / trends |
| POST | `/qa/score` | QA / compliance scoring |
| POST | `/alerts` | Per-call or aggregate alerts |

Shared flags on Fas 4 bodies: `reanalyze`, `use_mistral_llm`, `deep_analysis`, `llm_model`, `profile`.

- `reanalyze: false` (default) — reuse per-call report cache before aggregate steps
- `reanalyze: true` — force fresh `analyze_segments` for every call

### Edge AI (offline)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/edge/analyze-text` | Offline sentiment + intent on a single text string |
| POST | `/edge/analyze-segments` | Offline analysis on pre-transcribed segments (PII redaction for `callcenter`) |

See `docs/EDGE_AI.md`. No external LLM or Fas 4 enrichment.

### Observability & alerting

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/status/processes` | Recent pipeline/transcription status events (filterable by `job_id`, `component`) |
| GET | `/status/jobs/{job_id}` | Derived job status from event ring buffer |
| GET | `/status/health/detail` | Detailed health (registered analyzers, ASR backends, cache) |
| GET | `/alerting/status` | Webhook + circuit breaker status for dashboard/ops |
| POST | `/alerting/reset-circuit-breaker` | Reset webhook circuit breaker |

### Scan

- `POST /scan_process` — incremental directory scan (`transcribe` or `analyze_conversation`), optional `state_file`, `use_full_pipeline` for analyze operation

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

Aggregates cached agent metrics over one or more calls. Path `agent_id` must match body `agent_id`.

**Auth:** `X-API-Key` when `SENTIMENT_API_KEY` is set. **Rate limit:** `API_RATE_LIMIT_RPM` per client IP.

```bash
curl -X POST http://localhost:8000/agent_performance/Agent-42 \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret" \
  -d '{
    "agent_id": "Agent-42",
    "segments_list": [[
      {"text": "Hej och välkommen.", "speaker": "agent", "start": 0, "end": 2},
      {"text": "Fakturan är fel.", "speaker": "customer", "start": 2, "end": 5}
    ]],
    "window": "7d",
    "reanalyze": false,
    "deep_analysis": false
  }'
```

**Response (200):**

```json
{
  "agent_id": "Agent-42",
  "metrics": {
    "call_count": 1,
    "averages": {
      "empathy_score": 0.75,
      "talk_ratio": 0.45,
      "lexical_formality": 0.6,
      "de_escalation_effectiveness": 0.7
    },
    "trend_empathy": "stable",
    "computed_at": "2026-06-19T12:00:00"
  },
  "cached": false,
  "timestamp": "2026-06-19T12:00:01.000000+00:00"
}
```

`cached: true` on repeat requests with identical input (aggregate pre-computation).

---

### Semantic search

Hybrid vector + keyword search over provided calls. Max 50 calls, query max 500 chars.

```bash
curl -X POST http://localhost:8000/search/semantic \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret" \
  -d '{
    "query": "kunder klagade på faktura och låg empati",
    "top_k": 5,
    "segments_list": [[
      {"text": "Jag är arg på fakturan!", "speaker": "customer"},
      {"text": "Okej.", "speaker": "agent"}
    ]],
    "filters": {"agent": "Agent-42"}
  }'
```

**Response (200):**

```json
{
  "query": "kunder klagade på faktura och låg empati",
  "hits": [
    {
      "id": "0",
      "score": 0.42,
      "highlights": ["Jag är arg på fakturan!"],
      "metadata": {},
      "evidence_spans": [{"text": "Jag är arg på fakturan!"}]
    }
  ],
  "meta": {"num_docs": 1, "used_vector": false, "used_keyword": true},
  "timestamp": "2026-06-19T12:00:01.000000+00:00"
}
```

---

### Hot topics

Aggregated hot topics and trends over multiple calls (cached per window).

```bash
curl -X POST http://localhost:8000/insights/hot_topics \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret" \
  -d '{
    "segments_list": [
      [{"text": "Fakturan är fel", "speaker": "customer"}],
      [{"text": "Faktura stämmer inte igen", "speaker": "customer"}]
    ],
    "window": "7d"
  }'
```

**Response (200):**

```json
{
  "hot_topics": [
    {
      "topic": "faktura",
      "volume": 2,
      "avg_sentiment": -0.6,
      "trend": "rising",
      "evidence_spans": [{"text": "Fakturan är fel"}]
    }
  ],
  "meta": {"window": "7d", "n_calls": 2},
  "timestamp": "2026-06-19T12:00:01.000000+00:00"
}
```

---

### QA score

Run compliance QA scorecard on a single call's segments (max 200 segments).

```bash
curl -X POST http://localhost:8000/qa/score \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret" \
  -d '{
    "segments": [
      {"text": "Hej och välkommen till kundtjänst.", "speaker": "agent"},
      {"text": "Min faktura är fel.", "speaker": "customer"},
      {"text": "Jag förstår och beklagar besväret.", "speaker": "agent"}
    ],
    "profile": "callcenter",
    "use_mistral_llm": false
  }'
```

**Response (200):**

```json
{
  "qa": {
    "scorecard_name": "standard_support_v1",
    "overall_qa_score": 82.5,
    "passed": true,
    "risk_level": "low",
    "passed_criteria": ["greeting", "empathy"],
    "failed_criteria": [],
    "criteria_results": [],
    "computed_at": "2026-06-19T12:00:00"
  },
  "timestamp": "2026-06-19T12:00:01.000000+00:00"
}
```

---

### Alerts

Per-call alerts from pipeline results, or aggregate trend alerts.

```bash
# Per-call alerts
curl -X POST http://localhost:8000/alerts \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret" \
  -d '{
    "segments_list": [[
      {"text": "Jag är extremt arg!", "speaker": "customer"},
      {"text": "Okej.", "speaker": "agent"}
    ]]
  }'

# Aggregate trend alerts
curl -X POST http://localhost:8000/alerts \
  -H "Content-Type: application/json" \
  -d '{"aggregate": {"team_avg_empathy": 0.3, "hot_topic": "faktura"}}'
```

**Response (200):**

```json
{
  "alerts": [
    {
      "rule_id": "low_empathy",
      "severity": "high",
      "message": "Agent empathy below threshold",
      "evidence_spans": [{"text": "Okej."}],
      "recommended_actions": ["flag_supervisor", "create_coaching_task"]
    }
  ],
  "timestamp": "2026-06-19T12:00:01.000000+00:00"
}
```

---

### Rate limiting

When `API_RATE_LIMIT_RPM` > 0, excess requests return **429** with `error_code: rate_limit_exceeded`. Health checks are not rate-limited.

### Python client

```python
import httpx

client = httpx.Client(base_url="http://localhost:8000", headers={"X-API-Key": "secret"})
r = client.post("/analyze", json={"texts": ["Bra service!"]})
print(r.json()["results"])
```

---

## Errors

| HTTP | `error_code` | Meaning |
|------|----------------|---------|
| 401 | `unauthorized` | Missing/invalid `X-API-Key` |
| 409 | `conflict` | Job already cancelled / already finished |
| 413 | `payload_too_large` | Upload exceeded `API_MAX_UPLOAD_SIZE_MB` |
| 422 | `validation_error` | Pydantic validation or `ConfigurationError` |
| 429 | `rate_limit_exceeded` | `API_RATE_LIMIT_RPM` exceeded |
| 500 | `internal_error` / domain codes | Sanitized `detail`; domain errors include e.g. `transcription_failed`, `analysis_failed` |
| 502 | `llm_request_failed` | OpenRouter/Mistral failure |

All error JSON bodies include backward-compatible `detail` plus `request_id` (matches `X-Request-ID` header) and `error_code`.

---

## Testing

```bash
make test-api   # full API suite + ≥90% coverage on src/api
```

Coverage gates: CI `test` ≥ **80 %** on `src/`; `pyproject.toml` ≥ **85 %** on `src/`; `make test-api` / CI `api-test` ≥ **90 %** on `src/api`. See `docs/DEVELOPMENT.md` § Testing for the L0–L9 runbook.