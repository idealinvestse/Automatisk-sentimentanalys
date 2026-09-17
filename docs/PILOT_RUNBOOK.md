# Pilot Runbook — Conditional go

**Skapad:** 2026-07-17  
**Uppdaterad:** 2026-09-17 (beslutsregister + driftlägen)  
**Källa:** [DECISION_REPORT_2026-07-17.md](DECISION_REPORT_2026-07-17.md), [STRATEGY.md](../STRATEGY.md)  
**Syfte:** Operativ checklista för kontrollerad kundpilot utan okvalificerad production-pitch.  
**Ägare:** Oscar Delerud (korpus, annotering, kvalitetsgodkännande, go/no-go, kundpåståenden)

---

## 0. Beslutsregister och driftlägen (2026-09-17)

### 0.1 Beslutsregister

Alla beslut ägs och fattas av **Oscar Delerud**.

| ID | Beslut | Status | Godkännandepunkt / blockerad aktivering |
|----|-------|--------|----------------------------------------|
| R01 | Pilotmiljö: Windows 11, CUDA, NVIDIA RTX 5070 12 GB VRAM | **Beslutad** | Hårdvara fastställd. Numeriska kapacitetsmål (latens, RTF, volym) beslutas efter teknisk baslinjemätning. |
| R02 | Primär nytta: QA-stöd och coachning. Automatisering omfattar analys, bedömningar och rekommendationer — **inte** verkställande externa åtgärder | **Beslutad** | Inga externa webhook-/åtgärdsvägar aktiveras i piloten. |
| R03 | Kund-ID i ljudfilens **originalfilnamn** identifierar **kundorganisationen** som styr konfiguration | **Beslutad (princip)** | Blockerad för skarp import: filnamnsformat (R04) och faktiska kundmappningar saknas. |
| R04 | Filnamnets konkreta format beslutas senare | **Öppet** | Blockerar skarp filidentifiering — inte dokumentation eller syntetiska resolvertester. |
| R05 | Extern bearbetning får omfatta **text och råljud** när kundprofilen tillåter det, förutsatt lokal lagring av underlag och resultat | **Beslutad (princip)** | Blockerad per leverantör: vald provider, dataskyddsläge, försökstak och budget kräver uttrycklig kundprofil + Oscars aktivering. Lokal kopia ≠ leverantören saknar kopia. |
| R06 | Moln-STT som reserv efter lokal ASR väljs **per kundprofil** | **Beslutad (princip)** | Blockerad per kund: uttrycklig aktivering i kundprofilen krävs. Ingen generell automatisk molnreserv. |
| R07 | Lokal/extern analysordning väljs **per kundprofil** | **Beslutad (princip)** | Blockerad per kund: första kundprofilens modeller/providerkedja måste konfigureras. |
| R08 | Oscar Delerud äger korpus, annotering, kvalitetsgodkännande, go/no-go och godkännande av kundpåståenden | **Beslutad** | Löpande. Ingen annan part får godkänna kundkvalitet. |
| R09 | Leveransen omfattar dokumentation **och** tekniskt genomförande av nödvändiga nya funktioner | **Beslutad** | Genomförs etappvis enligt godkänd plan; varje etapp verifieras innan nästa. |

### 0.2 Två driftlägen

| Läge | Beskrivning | Styrande policy |
|------|-------------|-----------------|
| **A. Operatörspilot** (ny, aktiv utveckling) | Lokal drift på Oscars Windows 11/RTX 5070-dator. Kundorganisation identifieras från filnamn och styr bearbetning enligt beslutsregistret ovan. | Beslutsregistret (§0.1) + kundprofilens konfiguration. Extern bearbetning endast efter uttrycklig kundprofil + aktivering (R05–R07). |
| **B. Kundpilot** (befintlig, låst) | Kontrollerad kundpilot enligt beslutsrapporten 2026-07-17: lokal ASR, anonymize-före-LLM, Groq avstängd. | §1 Policy nedan + `verify_pilot_policy.py`. Gäller fortfarande för kundfacing drift utöver operatörspiloten. |

Läge A och B är separata driftformer. Att ett läge tillåter en handling betyder inte att det andra gör det.

### 0.3 Spårbarhetsmatris — granskningsytor → dokumentation

| # | Yta | Primär dokumentation | Status |
|---|-----|----------------------|--------|
| 1 | Produktmandat | STRATEGY.md | Uppdaterad 2026-09 |
| 2 | Användningsfall | STRATEGY.md, docs/PILOT_ONE_PAGER.md | Uppdaterad |
| 3 | Begrepp och mätetal | docs/ANALYZER_STRATEGY.md | Delvis — mätdefinitioner konsolideras i Etapp 7 |
| 4 | Systemgränser | docs/ARCHITECTURE.md | Uppdaterad med as-is/to-be |
| 5 | Huvudflöden | docs/ARCHITECTURE.md | Uppdaterad |
| 6 | Analyskatalog | docs/ANALYZER_STRATEGY.md | Befintlig tiers-modell |
| 7 | AI-/modellstyrning | docs/MULTI_PROVIDER_LLM.md, docs/LM_STUDIO_LOCAL.md | Uppdaterad; kundstyrd routing policy är to-be (Etapp 4) |
| 8 | Konfiguration | docs/WINDOWS_INSTALL.md + `src/install/config_schema.py` + `src/customers.py` | Etapp 2 landad (resolver + kläm). Default `mode: disabled` tills R04. |
| 9 | Data och proveniens | SECURITY.md, docs/DATA_01_CORPUS_SPEC.md, `src/api/call_store.py` | Etapp 3 landad: server-call-id, transkript före LLM, fingerprint i store. |
| 10 | Integritet och juridik | SECURITY.md, denna runbook | Uppdaterad; DPIA/avtalsläge för valda leverantörer är öppet (R05) |
| 11 | Säkerhet och åtkomst | SECURITY.md | Uppdaterad; användaridentitet/fleranvändaråtkomst är öppen fråga |
| 12 | Persistens och återställning | docs/PRODUCTION_CHECKLIST.md | Uppdaterad; RPO/RTO ej satta |
| 13 | Vetenskaplig validering | docs/DATA_01_CORPUS_SPEC.md, docs/DEVELOPMENT.md | DATA-01 levereras externt av Oscar |
| 14 | Test- och releasebevis | docs/DEVELOPMENT.md, docs/PRODUCTION_CHECKLIST.md | Gatestatus (PASS/FAIL/SKIP/BLOCKED/NOT RUN) separeras i Etapp 7 |
| 15 | Drift och kapacitet | docs/PRODUCTION_CHECKLIST.md, docs/WINDOWS_INSTALL.md | Kapacitetsmål ej satta (R01) |
| 16 | UX och tillit | docs/FE_BE_HARMONY_2026-07-17.md | Kundbadge på transkriberingssidan landad; dashboard-polish (Etapp 6) kvar |
| 17 | Integration och distribution | README.md, docs/WINDOWS_INSTALL.md | Befintlig |
| 18 | Leverans- och dokumentstyrning | Denna runbook §0 | Uppdaterad |

---

## 1. Policy (låst) — gäller Läge B: Kundpilot

> Dessa regler gäller den kundfacing piloten (Läge B). Operatörspiloten (Läge A) styrs av beslutsregistret §0.1 och respektive kundprofil — men kan aldrig vara *mindre* skyddad än detta golv vid kunddatahantering.

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

## 3. Rekommenderad `.env` för pilot (Läge B: Kundpilot)

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

**Prefererad:** Next.js BFF-proxy så API-nyckeln stannar server-side. Spegla **inte** nyckeln till `NEXT_PUBLIC_API_KEY`.

```bash
# webui/.env.local
NEXT_PUBLIC_USE_API_PROXY=1
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000   # behövs för WebSocket
SENTIMENT_API_BASE_URL=http://localhost:8000
SENTIMENT_API_KEY=<samma-som-backend>
```

**Legacy trusted-LAN / LM Studio-labb:** nyckel synlig i browser-bundle. Krävs för `NEXT_PUBLIC_USE_DIRECT_API=1` (jobs-panelen). **Inte** kundpilot.

```bash
NEXT_PUBLIC_USE_DIRECT_API=1
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
NEXT_PUBLIC_API_KEY=<samma-som-SENTIMENT_API_KEY>
# SENTIMENT_PILOT_ALLOW_LMSTUDIO=1   # bara isolerad labb + verify_pilot_policy --strict
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
