# Produktionschecklista

**Skapad:** 2026-06-28 (audit DOC-02)  
**Syfte:** Checklista innan produktionsdrift av Swedish Sentiment API och NiceGUI-dashboard.

---

## 1. Observability

- [ ] **Strukturerad loggning** — JSON/log aggregation (ELK, Loki, CloudWatch) med `request_id` från `X-Request-ID`
- [ ] **Health** — `GET /health` returnerar `{"status":"ok"}` (används av Docker healthcheck)
- [ ] **Metrics** — `GET /metrics` (Prometheus, **ingen API-nyckel** — begränsa via nätverk/firewall)
- [ ] **Scrape-config** — exempel:

```yaml
scrape_configs:
  - job_name: sentiment-api
    metrics_path: /metrics
    static_configs:
      - targets: ["api:8000"]
```

- [ ] **Tracing** (valfritt v0.5+) — OpenTelemetry för långa pipeline-anrop och ASR-jobb

---

## 2. Secrets

- [ ] **Miljövariabler** — sätt i deployment, aldrig i git:
  - `OPENROUTER_API_KEY` / `MISTRAL_API_KEY`
  - `GROQ_API_KEY`
  - `SENTIMENT_API_KEY` (API auth)
  - `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` (diarization)
- [ ] **Windows keyring** — `[install]` extra (`pyyaml`, `keyring`) för launcher secrets
- [ ] **`.env`** — i `.gitignore`; använd `.env.example` som mall utan riktiga värden
- [ ] **PII** — aktivera early redaction för `callcenter`-profil; granska LLM-routing (Groq GDPR gate)

---

## 3. GPU Docker

- [ ] **CUDA base image** — byt `FROM python:3.11-slim` till NVIDIA CUDA runtime vid GPU-behov
- [ ] **Kör med GPU** — `docker run --gpus all ...`
- [ ] **Volumes** — montera `HF_HOME` / `/cache/hf` för modellcache
- [ ] **Torch CUDA** — installera rätt wheel efter base image (se `scripts/dev-setup.ps1 -Cuda`)

Exempel (efter CUDA Dockerfile):

```bash
docker build -t sentimentanalys-gpu -f Dockerfile.gpu .
docker run --gpus all -p 8000:8000 -v hf_cache:/cache/hf sentimentanalys-gpu
```

---

## 4. Metrics (Prometheus)

Exponerade gauges (v0.5 grund):

| Metric | Typ | Beskrivning |
|--------|-----|-------------|
| `alerting_circuit_breaker_open` | Gauge | 1 = webhook circuit breaker öppen |
| `alerting_consecutive_failures` | Gauge | Antal på varandra följande webhook-fel |
| `sentiment_api_info{version="..."}` | Gauge | Statisk build-info (alltid 1) |

**AlertingState** — persisteras i `.cache/alerting_state.json`; synkas till gauges vid `/metrics`-scrape och vid state-ändringar.

**Framtida (PROD-01):** HTTP request counters, pipeline latency histogram, cache hit rate.

---

## 5. Drift & skalning

- [ ] **Rate limiting** — `SENTIMENT_RATE_LIMIT_RPM` i API settings
- [ ] **Redis cache** — `use_redis_cache` för multi-worker aggregate cache
- [ ] **Backup** — `outputs/`, `.cache/alerting_state.json`, användarkonfiguration
- [ ] **CI gate** — `pytest tests/test_api.py` med `--cov-fail-under=90` på `src/api`

---

## Relaterade dokument

- [SECURITY.md](../SECURITY.md)
- [docs/API.md](API.md)
- [docs/ROADMAP.md](ROADMAP.md)
- [CONTRIBUTING.md](../CONTRIBUTING.md)
