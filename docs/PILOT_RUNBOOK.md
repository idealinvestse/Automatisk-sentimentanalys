# Pilot Runbook — Conditional go

**Skapad:** 2026-07-17  
**Källa:** [DECISION_REPORT_2026-07-17.md](DECISION_REPORT_2026-07-17.md), [STRATEGY.md](../STRATEGY.md)  
**Syfte:** Operativ checklista för kontrollerad kundpilot utan okvalificerad production-pitch.

---

## 1. Policy (låst)

| Yta | Pilot/prod-regel |
|-----|------------------|
| ASR | **Endast lokal** (`provider=local`). Deepgram/cloud avstängt för PII-samtal. |
| LLM | OpenRouter → Mistral (EU/ZDR). `anonymize_before_llm=True` (callcenter-default). |
| Groq | **Förbjuden** för kunddata. Dev/lab endast. |
| Multi-worker live WS | Undvik tills Redis pub/sub för event hub finns. |
| Kvalitetskommunikation | Inga kundlöften om WER/F1 utan DATA-01 + mätning på riktig telefoni. |

Detaljerad motivering: decision pack §4–5.

---

## 2. Snabb verifiering av policy

Från repo-roten (`Automatisk-sentimentanalys/`):

```bash
# Ready-made template: copy .env.pilot.example → .env and fill secrets

# Kräver .env med API_PRODUCTION=true för full prod-gate-check
python scripts/verify_pilot_policy.py

# Striktare: faila om Groq-nyckel finns samtidigt som production
python scripts/verify_pilot_policy.py --strict

# Orchestrate policy + L7 (+ optional L8/L9):
python scripts/run_pilot_gates.py --strict --skip-l8 --skip-l9 --device cpu
```

Skriptet kontrollerar bl.a.:

- `API_PRODUCTION` / auth / media root när production-läge anges
- callcenter-profilens `anonymize_before_llm`
- att ASR-default i install-schema är `local`
- varningar om `GROQ_API_KEY` / `DEEPGRAM_API_KEY` i pilotläge

---

## 3. Rekommenderad `.env` för pilot

```bash
API_PRODUCTION=true
API_REQUIRE_AUTH=true
API_REQUIRE_MEDIA_ROOT=true
API_MEDIA_ROOT=/var/sentiment/media   # eller Windows-ekvivalent
SENTIMENT_API_KEY=<stark-nyckel>

OPENROUTER_API_KEY=<nyckel>
# GROQ_API_KEY=          # lämna tom / bortkommenterad
# DEEPGRAM_API_KEY=      # lämna tom / bortkommenterad

SENTIMENT_JSON_LOGS=1
OTEL_ENABLED=true
# OTEL_EXPORTER_OTLP_ENDPOINT=https://...
```

Profil: `callcenter` (PII-redaction på). Starta API med staging-compose eller GPU-image enligt [PRODUCTION_CHECKLIST.md](PRODUCTION_CHECKLIST.md).

### Webui auth (H1)

**Prefererad:** Next.js BFF-proxy så API-nyckeln stannar server-side:

```bash
# webui/.env.local
NEXT_PUBLIC_USE_API_PROXY=1
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000   # behövs för WebSocket
SENTIMENT_API_BASE_URL=http://localhost:8000
SENTIMENT_API_KEY=<samma-som-backend>
```

**Legacy trusted-LAN:** nyckel synlig i browser-bundle:

```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
NEXT_PUBLIC_API_KEY=<samma-som-SENTIMENT_API_KEY>
```

Se `webui/.env.production.example` och [FE_BE_HARMONY_2026-07-17.md](FE_BE_HARMONY_2026-07-17.md).

### Multi-worker / live WS

När `uvicorn --workers > 1` **måste** Redis vara på:

```bash
API_USE_REDIS_CACHE=true
REDIS_URL=redis://localhost:6379/0
```

Annars är jobs/tickets/WS event hub process-lokala. `/status/health/detail` och `/ready` exponerar Redis/hub-backend.

---

## 4. Release-gates före pilot (L7–L9)

Kör och bocka i PRODUCTION_CHECKLIST:

| Gate | Kommando / handling |
|------|---------------------|
| L7 ASR | `python -m src.evaluate audio smoke --device cuda` (eller `cpu`) |
| L8 LLM | `python -m src.evaluate llm-quality` (om deep-path på) |
| L9 Staging | `docker compose -f docker-compose.staging.yml up` + `python scripts/staging_observability_smoke.py` |
| Webui live | Manuell pass `/testlab` mot riktig API |
| Spotcheck | En svensk call via CLI `analyze-call` + samma i dashboard |

---

## 5. DATA-01 före kvalitetsclaim

Se [DATA_01_CORPUS_SPEC.md](DATA_01_CORPUS_SPEC.md).

```bash
# CI / lokal path-övning (syntetisk — ersätter inte riktig telefoni):
python scripts/generate_pilot_corpus.py --import --pilot-gate

# Riktig anonymiserad korpus (externt):
python scripts/import_domain_corpus.py --source-dir /secure/anonymized --pilot-gate
python scripts/evaluate_real_corpus.py \
  --sentiment-csv data/import/callcenter_val_real.csv \
  --intent-jsonl data/import/intent_val_real.jsonl
```

`--pilot-gate` kräver minst **500** sentiment-rader och **200** intent-rader (decision pack).
Kvalitetslöften till kund kräver **riktig** telefoni-slice, inte bara den syntetiska bundlen.

---

## 6. Kundkommunikation

Använd [PILOT_ONE_PAGER.md](PILOT_ONE_PAGER.md). Lova man inte:

- “enterprise suite-paritet” med NICE/Genesys
- telemetri-/WER-siffror utan er uppmätta telefoni-slice
- cloud-STT som default

---

## 7. Relaterat

- [SECURITY.md](../SECURITY.md) — cloud STT / LLM-egress
- [DEVELOPMENT.md](DEVELOPMENT.md) — DATA-01 import (befintlig runbook)
- [PRODUCTION_CHECKLIST.md](PRODUCTION_CHECKLIST.md) — infra + L7–L9
- [reports/parallel-research/](../reports/parallel-research/) — externt underlag
