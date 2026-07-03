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

- [x] **Miljövariabler** — `docker-compose.webui.yml` använder `env_file: .env` för API-service. Se `.env.example` för alla env vars med beskrivningar. Variabler som måste sättas i deployment:
  - `OPENROUTER_API_KEY` / `MISTRAL_API_KEY` — LLM-providers
  - `GROQ_API_KEY` — Groq (US/Saudi — GDPR gate kräver `groq_eu_residency=true`)
  - `SENTIMENT_API_KEY` — API auth
  - `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` — diarization (pyannote)
- [x] **Production guards** (v0.5) — `src/api/settings.py:99-101` + `validate_production_settings()`:
  - `API_PRODUCTION=true` — kräver auth + media root
  - `API_REQUIRE_AUTH=true` — kräver `SENTIMENT_API_KEY`
  - `API_REQUIRE_MEDIA_ROOT=true` — kräver `API_MEDIA_ROOT`
- [ ] **Windows keyring** — `[install]` extra (`pyyaml`, `keyring`) för launcher secrets
- [x] **`.env`** — i `.gitignore` (rad 23). `.env.example` skapad 2026-07-03 som mall.
- [x] **PII** — early redaction aktiverad som default för `callcenter`-profil (`anonymize_before_llm: True` i `src/profiles.py:166`, ändrad 2026-07-03). PII redakteras före både lokal analys och LLM-anrop. Groq GDPR gate implementerad (`src/llm/groq_client.py:282`).

---

## 3. GPU Docker

- [x] **CUDA Dockerfile** — `Dockerfile.gpu` (NVIDIA CUDA 12.1 + cuDNN 8, Ubuntu 22.04). Torch installeras från `--index-url https://download.pytorch.org/whl/cu121`.
- [ ] **Kör med GPU** — `docker run --gpus all ...` (verifiera på riktig GPU-host, se steg nedan)
- [x] **Volumes** — `HF_HOME=/cache/hf` i `Dockerfile.gpu`. `docker-compose.webui.yml` monterar `hf_cache:/cache/hf`.
- [x] **Torch CUDA** — installeras i `Dockerfile.gpu` (rad 30).

```bash
docker build -t sentimentanalys-gpu -f Dockerfile.gpu .
docker run --gpus all -p 8000:8000 -v hf_cache:/cache/hf sentimentanalys-gpu
```

### GPU-verifieringssteg (utför på riktig GPU-host innan produktion)

1. **Bygg image:**
   ```bash
   docker build -t sentimentanalys-gpu -f Dockerfile.gpu .
   ```

2. **Verifiera CUDA-tillgänglighet inuti containern:**
   ```bash
   docker run --gpus all --rm sentimentanalys-gpu python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
   ```
   Förväntad output: `CUDA available: True` + GPU-namn (t.ex. `NVIDIA GeForce RTX 4090`).

3. **Kör API med GPU och env-file:**
   ```bash
   docker run --gpus all -p 8000:8000 \
     --env-file .env \
     -v hf_cache:/cache/hf \
     -v $(pwd)/outputs:/app/outputs \
     sentimentanalys-gpu
   ```

4. **Verifiera sentiment-modell laddas på GPU:**
   ```bash
   curl -X POST http://localhost:8000/analyze \
     -H "Content-Type: application/json" \
     -d '{"texts":["Tack för hjälpen!"]}'
   ```
   Förväntat: snabbare respons än CPU (typiskt <500ms vs 2-5s).

5. **Verifiera ASR (faster-whisper) på GPU:**
   ```bash
   curl -X POST http://localhost:8000/transcribe \
     -H "Content-Type: application/json" \
     -d '{"audio_path":"samples/audio/sv/demo.wav"}'
   ```
   Förväntat: transkription på några sekunder (vs 30s+ på CPU).

6. **Kontrollera GPU-användning under last:**
   ```bash
   docker exec -it <container_id> nvidia-smi
   ```
   Förväntat: Python-process synlig med GPU-minnesanvändning.

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
- [x] **Backup** — `scripts/backup.py` skapar timestampade tar.gz-arkiv med rotation. Se "Backup-guide" nedan för cron-exempel.
- [x] **CI gate** — `.github/workflows/ci.yml` jobb `api-test` (rad 65-88) kör `pytest tests/test_api.py` med `--cov-fail-under=90` på `src/api`. Separat `webui`-jobb (rad 99) kör lint+build+e2e.

### Backup-guide

`scripts/backup.py` skapar timestampade tar.gz-arkiv med automatisk rotation.

```bash
# Manuell backup (default: /backups, behåll 7 arkiv)
python scripts/backup.py --output-dir /backups --keep 7

# Med Redis BGSAVE (om API_USE_REDIS_CACHE=true)
python scripts/backup.py --output-dir /backups --keep 7 --redis
```

Cron-exempel (daglig backup kl 02:00):
```cron
0 2 * * * cd /app && python scripts/backup.py --output-dir /backups --keep 7 --redis >> /var/log/backup.log 2>&1
```

Följande data backas upp:

| Path | Innehåll | Frekvens |
|------|----------|----------|
| `outputs/` | Pipeline-rapporter, utvärderingsresultat | Daglig |
| `.cache/alerting_state.json` | Persistent alerting-tillstånd (circuit breaker, webhook-retries) | Daglig |
| `.cache/aggregates/` | Pre-computed aggregate cache | Kan återskapas — backup valfritt |
| `configs/` | LLM-config, QA-scorecards, alerting-config, profiler | Vid ändring (git-tracked) |
| `data/` | Träningsdata, lexicon, model catalog | Vid ändring (git-tracked) |

---

## 6. Verifieringsresultat (2026-07-03)

| Kategori | Pass | Fail | Partial |
|----------|------|------|---------|
| Observability | 6 | 0 | 0 |
| Secrets | 5 | 0 | 0 |
| GPU Docker | 3 | 1 | 0 |
| Drift & skalning | 4 | 0 | 0 |
| **Total** | **18** | **1** | **0** |

### Kvarstående gaps (måste åtgärdas före produktion):

1. **GPU-verifiering** — `Dockerfile.gpu` är byggt men ej testat på riktig GPU-host. Se "GPU-verifieringssteg" ovan för steg-för-steg-guide.

### Åtgärdade i denna session:

- ✅ `.env.example` skapad med alla 25+ env vars
- ✅ `docs/examples/prometheus.yml` scrape-config-exempel
- ✅ Checklistan verifierad mot faktisk kodbase (alla items har fil:line-referenser)
- ✅ `scripts/backup.py` automatiserad backup-script med rotation + cron-exempel
- ✅ Metrics-auth-beteende dokumenterat (kräver API-nyckel när auth aktiverat)
- ✅ PII-default ändrad: `anonymize_before_llm=True` för callcenter-profil (production-safe)
- ✅ `docker-compose.webui.yml` uppdaterad med `env_file: .env` + `api_cache` volume
- ✅ GPU-verifieringssteg dokumenterade (6 steg från build till nvidia-smi)

---

## Relaterade dokument

- [SECURITY.md](../SECURITY.md)
- [docs/API.md](API.md)
- [docs/ROADMAP.md](ROADMAP.md)
- [CONTRIBUTING.md](../CONTRIBUTING.md)
- [.env.example](../.env.example) — miljövariabel-mall
- [docs/examples/prometheus.yml](examples/prometheus.yml) — scrape-config-exempel
