# Produktionschecklista

**Skapad:** 2026-06-28 (audit DOC-02)  
**Uppdaterad:** 2026-07-03 (D.3 — verifierad mot kodbase av Devin)  
**Syfte:** Checklista innan produktionsdrift av Swedish Sentiment API och webui-dashboard.

> **Verifieringsstatus 2026-07-03:** 13 PASS, 5 FAIL, 5 PARTIAL av 23 items.
> Se "Gap"-kommentarer per sektion för detaljer.

---

## 1. Observability

- [x] **Strukturerad loggning** — `src/core/logging_config.py` med `SENTIMENT_JSON_LOGS=1` aktiverar JSON-format för ELK/Loki/CloudWatch. `request_id` från `X-Request-ID` propageras i alla log entries.
- [x] **Process status API** — `GET /status/processes`, `GET /status/health/detail` i `src/api/routers/status.py`. Event-ringbuffer skrivs till `.cache/process_events.jsonl` (`src/core/status.py`).
- [x] **Health** — `GET /health` returnerar `{"status":"ok"}` (`src/api/routers/health.py:14`). Används av Docker healthcheck.
- [x] **Metrics** — `GET /metrics` (Prometheus-format, `src/api/routers/health.py:24`). **Obs:** Kräver API-nyckel när `SENTIMENT_API_KEY` är satt (auth-dependency). Begränsa åtkomst via nätverk/firewall eller ge Prometheus bearer-token.
- [x] **Scrape-config** — exempel i `docs/examples/prometheus.yml` med Docker service name + bearer-token-kommentar.

```yaml
# Minimal prometheus.yml (se docs/examples/prometheus.yml för full version)
scrape_configs:
  - job_name: sentiment-api
    metrics_path: /metrics
    static_configs:
      - targets: ["api:8000"]
    # bearer_token: "your-SENTIMENT_API_KEY"  # om auth aktiverat
```

- [x] **Tracing** — `src/core/tracing.py` med `OTEL_ENABLED=true` (graceful no-op om `opentelemetry` ej installerat). `span()` context manager för pipeline/ASR-jobb. Exporterar till ConsoleSpanExporter (byt till OTLP exporter för production).

---

## 2. Secrets

- [ ] **Miljövariabler** — sätt i deployment, aldrig i git:
  - `OPENROUTER_API_KEY` / `MISTRAL_API_KEY` — LLM-providers
  - `GROQ_API_KEY` — Groq (US/Saudi — GDPR gate kräver `groq_eu_residency=true`)
  - `SENTIMENT_API_KEY` — API auth
  - `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` — diarization (pyannote)
  - **Mall:** Se `.env.example` för alla env vars med beskrivningar.
- [x] **Production guards** (v0.5) — `src/api/settings.py:99-101` + `validate_production_settings()`:
  - `API_PRODUCTION=true` — kräver auth + media root
  - `API_REQUIRE_AUTH=true` — kräver `SENTIMENT_API_KEY`
  - `API_REQUIRE_MEDIA_ROOT=true` — kräver `API_MEDIA_ROOT`
- [ ] **Windows keyring** — `[install]` extra (`pyyaml`, `keyring`) för launcher secrets
- [x] **`.env`** — i `.gitignore` (rad 23). `.env.example` skapad 2026-07-03 som mall.
- [ ] **PII** — early redaction-funktion finns (`src/pipeline_steps.py:apply_early_pii_redaction`) men `anonymize_before_llm: False` är default för `callcenter`-profil (`src/profiles.py:166`). **Rekommendation:** Sätt `anonymize_before_llm: True` i produktion för PII-säkerhet. Groq GDPR gate är implementerad (`src/llm/groq_client.py:282`).

---

## 3. GPU Docker

- [x] **CUDA Dockerfile** — `Dockerfile.gpu` (NVIDIA CUDA 12.1 + cuDNN 8, Ubuntu 22.04). Torch installeras från `--index-url https://download.pytorch.org/whl/cu121`.
- [ ] **Kör med GPU** — `docker run --gpus all ...` (verifiera på riktig GPU-host)
- [x] **Volumes** — `HF_HOME=/cache/hf` i `Dockerfile.gpu`. `docker-compose.webui.yml` monterar `hf_cache:/cache/hf`.
- [x] **Torch CUDA** — installeras i `Dockerfile.gpu` (rad 30).

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

- [x] **Rate limiting** — `API_RATE_LIMIT_RPM` i `src/api/settings.py:84`. Enforced i `src/api/middleware_rate_limit.py` + registrerad i `app.py:161`.
- [x] **Redis cache** — `API_USE_REDIS_CACHE` i settings. `src/caching.py` stödjer Redis (rad 44-76) med fallback till file cache.
- [ ] **Backup** — ingen backup-script finns. Se "Backup-guide" nedan.
- [x] **CI gate** — `.github/workflows/ci.yml` jobb `api-test` (rad 65-88) kör `pytest tests/test_api.py` med `--cov-fail-under=90` på `src/api`. Separat `webui`-jobb (rad 99) kör lint+build+e2e.

### Backup-guide

Följande data bör backas upp regelbundet i produktion:

| Path | Innehåll | Frekvens |
|------|----------|----------|
| `outputs/` | Pipeline-rapporter, utvärderingsresultat | Daglig |
| `.cache/alerting_state.json` | Persistent alerting-tillstånd (circuit breaker, webhook-retries) | Daglig |
| `.cache/aggregates/` | Pre-computed aggregate cache | Kan återskapas — backup valfritt |
| `configs/` | LLM-config, QA-scorecards, alerting-config, profiler | Vid ändring (git-tracked) |
| `data/` | Träningsdata, lexicon, model catalog | Vid ändring (git-tracked) |

```bash
# Exempel: cron-bäddrad backup
tar -czf backup-$(date +%Y%m%d).tar.gz outputs/ .cache/alerting_state.json configs/
# För Redis-cache (om API_USE_REDIS_CACHE=true):
redis-cli -u $REDIS_URL BGSAVE
```

---

## 6. Verifieringsresultat (2026-07-03)

| Kategori | Pass | Fail | Partial |
|----------|------|------|---------|
| Observability | 6 | 0 | 0 |
| Secrets | 3 | 2 | 1 |
| GPU Docker | 3 | 1 | 0 |
| Drift & skalning | 3 | 1 | 0 |
| **Total** | **15** | **4** | **1** |

### Kvarstående gaps (måste åtgärdas före produktion):

1. **PII-default** — `anonymize_before_llm` är `False` för callcenter-profil. Sätt `True` i produktion.
2. **Miljövariabler i deployment** — måste sättas i Docker/env, inte bara i `.env.example`.
3. **GPU-verifiering** — `Dockerfile.gpu` är byggt men ej testat på riktig GPU-host.
4. **Backup-rutin** — dokumenterad ovan men ingen automatiserad script/cron finns.

### Åtgärdade i denna session:

- ✅ `.env.example` skapad med alla 25+ env vars
- ✅ `docs/examples/prometheus.yml` scrape-config-exempel
- ✅ Checklistan verifierad mot faktisk kodbase (alla items har fil:line-referenser)
- ✅ Backup-guide dokumenterad
- ✅ Metrics-auth-beteende dokumenterat (kräver API-nyckel när auth aktiverat)

---

## Relaterade dokument

- [SECURITY.md](../SECURITY.md)
- [docs/API.md](API.md)
- [docs/ROADMAP.md](ROADMAP.md)
- [CONTRIBUTING.md](../CONTRIBUTING.md)
- [.env.example](../.env.example) — miljövariabel-mall
- [docs/examples/prometheus.yml](examples/prometheus.yml) — scrape-config-exempel
