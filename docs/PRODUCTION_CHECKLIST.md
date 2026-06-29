# Produktionschecklista

**Skapad:** 2026-06-28 (audit DOC-02)  
**Uppdaterad:** 2026-06-28 (v0.5)  
**Syfte:** Checklista innan produktionsdrift av Swedish Sentiment API och NiceGUI-dashboard.

---

## 1. Observability

- [x] **Strukturerad loggning** — JSON/log aggregation (ELK, Loki, CloudWatch) med `request_id` från `X-Request-ID` — sätt `SENTIMENT_JSON_LOGS=1`
- [x] **Process status API** — `GET /status/processes`, `GET /status/health/detail` + `.cache/process_events.jsonl`
- [x] **Health** — `GET /health` returnerar `{"status":"ok"}` (används av Docker healthcheck)
- [x] **Metrics** — `GET /metrics` (Prometheus, **ingen API-nyckel** — begränsa via nätverk/firewall)
- [ ] **Scrape-config** — exempel:

```yaml
scrape_configs:
  - job_name: sentiment-api
    metrics_path: /metrics
    static_configs:
      - targets: ["api:8000"]
```

- [ ] **Tracing** (valfritt v0.5+) — OpenTelemetry för långa pipeline-anrop och ASR-jobb — sätt `OTEL_ENABLED=true`

---

## 2. Secrets

- [ ] **Miljövariabler** — sätt i deployment, aldrig i git:
  - `OPENROUTER_API_KEY` / `MISTRAL_API_KEY`
  - `GROQ_API_KEY`
  - `SENTIMENT_API_KEY` (API auth)
  - `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` (diarization)
- [ ] **Production guards** (v0.5):
  - `API_PRODUCTION=true` — kräver auth + media root
  - `API_REQUIRE_AUTH=true` — kräver `SENTIMENT_API_KEY`
  - `API_REQUIRE_MEDIA_ROOT=true` — kräver `API_MEDIA_ROOT`
- [ ] **Windows keyring** — `[install]` extra (`pyyaml`, `keyring`) för launcher secrets
- [ ] **`.env`** — i `.gitignore`; använd `.env.example` som mall utan riktiga värden
- [ ] **PII** — aktivera early redaction för `callcenter`-profil; granska LLM-routing (Groq GDPR gate)

---

## 3. GPU Docker

- [x] **CUDA Dockerfile** — `Dockerfile.gpu` (NVIDIA CUDA 12.1 runtime)
- [ ] **Kör med GPU** — `docker run --gpus all ...`
- [ ] **Volumes** — montera `HF_HOME` / `/cache/hf` för modellcache
- [ ] **Torch CUDA** — installeras i `Dockerfile.gpu`

```bash
docker build -t sentimentanalys-gpu -f Dockerfile.gpu .
docker run --gpus all -p 8000:8000 -v hf_cache:/cache/hf sentimentanalys-gpu
```

---

## 4. Metrics (Prometheus)

| Metric | Typ | Beskrivning |
|--------|-----|-------------|
| `alerting_circuit_breaker_open` | Gauge | 1 = webhook circuit breaker öppen |
| `alerting_consecutive_failures` | Gauge | Antal på varandra följande webhook-fel |
| `sentiment_api_info{version="..."}` | Gauge | Statisk build-info (alltid 1) |
| `http_requests_total` | Counter | HTTP requests per method/path/status |
| `http_request_duration_seconds` | Histogram | HTTP latency |
| `pipeline_duration_seconds` | Histogram | Pipeline end-to-end (v0.5) |
| `analyzer_duration_seconds` | Histogram | Per-analyzer timing (v0.5) |
| `llm_requests_total` | Counter | LLM calls per provider/model/outcome (v0.5) |
| `cache_operations_total` | Counter | Cache hit/miss (v0.5) |

**Framtida:** ASR job duration, LLM token/cost counters.

---

## 5. Drift & skalning

- [x] **Rate limiting** — `API_RATE_LIMIT_RPM` i API settings
- [ ] **Redis cache** — `API_USE_REDIS_CACHE` för multi-worker aggregate cache
- [ ] **Backup** — `outputs/`, `.cache/alerting_state.json`, användarkonfiguration
- [x] **CI gate** — `pytest tests/test_api.py` med `--cov-fail-under=90` på `src/api`

---

## Relaterade dokument

- [SECURITY.md](../SECURITY.md)
- [docs/API.md](API.md)
- [docs/ROADMAP.md](ROADMAP.md)
- [CONTRIBUTING.md](../CONTRIBUTING.md)
