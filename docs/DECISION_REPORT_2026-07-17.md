# Executive Decision Pack — Automatisk-sentimentanalys

**Datum:** 2026-07-17  
**Version underlag:** 0.5.0 (production-ready beta)  
**Syfte:** Underlag för pilot go/no-go, compliance-läge och 90-dagars prioritering  
**Metod:** Intern repo-genomlysning + fyra Parallel deep research (`pro-fast`)

---

## 1. Verdict

### Conditional go — kontrollerad kundpilot

Produktplattformen är tekniskt mogen nog för en **avgränsad pilot** (1–2 nordiska kontaktcenter / interna QA-team) under strikta villkor. Den är **inte** redo för okvalificerade produktionsanspråk om accuracy, 100 % QM eller “enterprise suite-paritet”.

| Dimension | Bedömning |
|-----------|-----------|
| Produkt / arkitektur | Stark: hybrid local-first, registry, CI, staging, Fas 4-backend |
| Kvalitetsbevis | Svag: sentiment ~66 % på *syntetisk* val; saknar verklig call-corpus |
| Compliance-story | Stark *om* local ASR + PII-redaction + EU LLM; svag om Groq/Deepgram används slarvigt |
| Go-to-market vs NICE/Genesys | Rätt wedge (svenska + EU-residency), fel race (full CCaaS-paritet) |
| Release-gates L7–L9 | Öppna — måste stängas före “production pilot”-etikett |

**Pilot tillåten när alla villkor nedan är uppfyllda.** Annars: **no-go** tills DATA-01 + L7–L9 är klara.

### Villkor för conditional go

1. **ASR:** endast lokal KB-Whisper / faster-whisper i pilot (Deepgram avstängt för PII-samtal).
2. **LLM:** OpenRouter → Mistral EU / ZDR-allow-list; `anonymize_before_llm=True` (callcenter-default).
3. **Groq:** förbjuden i pilot/prod med kunddata (dev-only OK).
4. **DATA-01:** minst ~500 manuellt granskade/anonymiserade samtal (eller motsv. segment) innan kvalitetskommunikation till kund.
5. **Release:** L7 ASR smoke, L8 LLM quality (om deep-path på), L9 staging smoke, manuell webui↔API via `/testlab`, spotcheck svensk call.
6. **Juridik:** DPIA + DPA med underleverantörer innan go-live; IVR/kundinformation om AI-analys.

---

## 2. Nuläge (intern genomlysning)

### Vad som är starkt

- Hybrid pipeline: lokal ASR → diarization → analyzer registry → selektiv LLM deep-path ([`docs/ROADMAP.md`](ROADMAP.md), [`docs/ANALYZER_STRATEGY.md`](ANALYZER_STRATEGY.md)).
- Fas 4: agent performance, QA scorecards, semantic search, hot topics, alerting.
- CI med coverage-gates; Docker staging (api + webui + redis + prometheus).
- Säkerhetsdokumentation för cloud STT och externa LLM-anrop ([`SECURITY.md`](../SECURITY.md)).
- Produktionschecklista: 19–20 PASS på infra/secrets/GPU-docs; kvarvarande gaps är medvetna ([`docs/PRODUCTION_CHECKLIST.md`](PRODUCTION_CHECKLIST.md)).

### Beslutsblockerare

| Gap | Evidens | Effekt |
|-----|---------|--------|
| Ingen riktig produktionskorpus | DATA-01 väntar; `data/import/` gitignored | Kan inte lova WER/F1 till kund |
| Sentiment under CI-tröskel på domain val | `reports/domain_baseline.json`: 0.664 acc / 0.664 F1 vs gate 0.70/0.68 | Risk för överdriven tillit |
| Intent = heuristic | `configs/analyzer_eval.yaml`: switch kräver +0.05 F1 | OK för pilot triage; inte “ML intent”-pitch |
| Fas 4 KPI stubs | `src/evaluate.py` | Coaching/hot-topic-siffror ej validerade |
| L7–L9 unchecked | PRODUCTION_CHECKLIST | Release-etikett saknar helstack-bevis |
| Multi-worker WS hub | process-lokal `TranscriptionEventHub` | Undvik multi-worker live-transkription i pilot |
| Ingen STRATEGY.md (före detta pack) | — | Positionering var implicit i ROADMAP |

### Arkitekturprinciper som redan är låsta i kod

- Local-first ASR; cloud STT opt-in only.
- Graceful degradation / `unavailable` när deep-path saknas.
- PII-first för callcenter-profil.

---

## 3. Externa insikter (Parallel R1–R4)

Fullständiga rapporter: [`reports/parallel-research/`](../reports/parallel-research/).

### R1 — Marknad & positionering

Källa: [nordic-cc-intelligence-market-2026.md](../reports/parallel-research/nordic-cc-intelligence-market-2026.md)

- Speech analytics globalt ~USD 2.8–3.8B (2025) med hög tillväxt; Norden är en högpenetrationsficka (bank, telco, försäkring, offentlig sektor).
- Köpare: (1) enterprise in-house CC, (2) BPO, (3) interna QA/compliance — olika triggers.
- **Feature parity är table stakes** hos NICE, Genesys, Verint+Calabrio, Observe.AI. Differentiering = språk, governance, residency, workflow-djup — inte “vi har också summarization”.
- SaaS-signaler: ~USD 71–160/agent/mån entry–mid; USD 150–300+ med premium AI. EU/Nordic ASR-premium ofta 20–40 % över engelska commodity-API:er.
- **Strategisk wedge:** hybrid local-first (svensk ASR + EU LLM + tunn control plane) — undvik att slåss mot CCaaS-buntar.

### R2 — GDPR / EU-residency

Källa: [gdpr-asr-llm-eu-residency-2026.md](../reports/parallel-research/gdpr-asr-llm-eu-residency-2026.md)

- Deepgram har EU-endpoint (GA jul 2025) men subprocessors kvarstår — EU-region ≠ automatiskt “säkert för Article 9”.
- OpenRouter: lås ZDR-allow-list; fail-closed.
- Mistral: default retention upp till 30 dagar utan no-retention/opt-out — kräver avtal + EU-pin.
- **Groq: US-only** — olämplig för Article 9 / känslig kundtjänst.
- HF *nedladdning* av vikter ≠ personuppgiftsöverföring; HF Inference kräver DPA.
- Article 9 / minderåriga / Art. 22-profilering → **local-only** är enda försvarbara vägen.
- IMY: DPIA före AI-profilering i kundtjänst; IVR-samtycke/transparens rekommenderas.

### R3 — Kostnad & SLO

Källa: [asr-llm-cost-slo-callcenter-2026.md](../reports/parallel-research/asr-llm-cost-slo-callcenter-2026.md)

- Deepgram Nova-3 ~USD 0.0065–0.0077/min; self-host faster-whisper ofta ~USD 0.0016–0.0033/min på commodity/A10-klass.
- Break-even self-host vs managed: ungefär **150–220k audio-min/mån** (inkl. ops-overhead); under det vinner API på TCO *om* residency tillåter.
- Batch post-call p95: tiotals sekunder/min fil på GPU; streaming agent-assist: hundratals ms STT + 1.2–2.5s e2e stack.
- LLM analytics per ~7 min call ofta **sub-cent till några cent** på små modeller — projektets budget `cost_budget_per_call: 0.08` ([`configs/llm_config.yaml`](../configs/llm_config.yaml)) är generös marginal.
- **Residency trump:ar kostnadsoptimering** för reglerade workloads.

### R4 — Svensk ASR / sentiment / intent gates

Källa: [swedish-asr-sentiment-eval-sota-2026.md](../reports/parallel-research/swedish-asr-sentiment-eval-sota-2026.md)

- KB-Whisper large-v3: ~4.1–5.4 % WER på rena benchmarks ([KBLab](https://kb-labb.github.io/posts/2025-03-07-welcome-KB-Whisper)) — rätt default (redan i produkten).
- Produktionstelefoni: förväntan 2–4× sämre än clean; realistiskt **10–15 % WER**-band för svensk first-pass; toppsystem på bra ljud 8–12 %.
- Syntetiska set överskattar 1.5–3× (ibland mer).
- Föreslagna **produktionsgates** (extern research): ≤10 % WER på representativ telefoni-slice, ≥85 % intent macro-F1, ≥80 % sentiment accuracy, på ≥500 riktiga samtal.
- Projektets heuristiska intent (~76.5 % F1) är nära CI-tröskel men under “prod pitch” 85 %; sentiment 66 % på synthetic är **under** både intern CI och externa prod-gates.

---

## 4. ce-pov — Domslut mot *detta* projekt

Varje verdict är Tier 3 (privacy/compliance) eller Tier 2 (arkitektur/kostnad), grundad i incumbents i repo + Parallel-externa fakta.

### 4.1 Deepgram cloud STT — **Hold / conditional opt-in**

| Floor | Faktum |
|-------|--------|
| Projekt | Default `asr.provider: local`; SECURITY.md: raw audio lämnar maskinen vid cloud; v1 opt-in Deepgram |
| Externt | EU-endpoint finns, men subprocessors + Article 9-risk kvarstår |

**Verdict:** Behåll opt-in för *icke-PII / demos / latency-experiment*. **Förbjud** för svensk kundtjänst-PII i pilot/prod tills DPA + EU-pin + masking-policy är signerade och testade. Local KB-Whisper är incumbenten — byt inte default.

### 4.2 Groq — **Reject for production customer data**

| Floor | Faktum |
|-------|--------|
| Projekt | `groq` i llm_config med explicit US/Saudi-varning; GDPR-gate i kod |
| Externt | US-only, ingen EU-hosting; ej för Article 9 |

**Verdict:** Dev/latency-lab endast. Ingen prod-routing av callcenter-transcripts.

### 4.3 OpenRouter → Mistral (selektiv LLM) — **Conditional adopt (keep)**

| Floor | Faktum |
|-------|--------|
| Projekt | Default deep-path via OpenRouter/Mistral; budget 0.08 USD/call; anonymize default true för callcenter i profiles |
| Externt | ZDR-allow-list krävs; Mistral retention kräver no-retention/EU-pin |

**Verdict:** Behåll som premium deep-path. Lås provider-allow-list + retention-avtal innan pilot. Fail-closed utan ZDR/EU.

### 4.4 Self-hosted GPU ASR (KB-Whisper) — **Adopt / reinforce**

| Floor | Faktum |
|-------|--------|
| Projekt | Default engine; `Dockerfile.gpu`; download-asr; roadmap svensk-first |
| Externt | KB-Whisper SOTA open Swedish; break-even ~150–220k min/mån; residency driver self-host |

**Verdict:** Förstärk som produktlöfte #1. Mät WER på *er* telefoni innan kundclaim. GPU-host + L7 är P0.

### 4.5 Intent model vs heuristic — **Hold switch**

| Floor | Faktum |
|-------|--------|
| Projekt | Heuristic default; `model_switch_min_f1_gain: 0.05` |
| Externt | Domän-finetune behövs för 85 %+ svensk intent |

**Verdict:** Byt inte backend förrän DATA-01 + träning slår heuristic med ≥0.05. Pitcha “regelbaserad intent + LLM-coaching” ärligt tills dess.

---

## 5. Beslutstabell (rekommenderat låst läge)

| Beslut | Rekommendation | Konsekvens om ni väntar / gör fel |
|--------|----------------|-----------------------------------|
| Pilot | **Conditional go** under §1-villkor | Okvalificerad “prod”-pitch → förtroende- och compliance-risk |
| Deployment | **Hybrid local-first**: lokal ASR + selektiv EU LLM | Cloud-heavy förstör wedge och DPIA |
| Deepgram | Opt-in, **aldrig default** för PII | Raw audio egress; svår IMY-förklaring |
| Groq | **Dev-only** | US/Saudi transfer-risk |
| OpenRouter/Mistral | Tillåten med ZDR + no-retention + anonymize | Retention/subprocessor-gap |
| DATA-01 | **P0**: ≥500 riktiga anonymiserade samtal | Kvalitetsgates och LoRA/intent blockerade |
| Intent | Heuristic tills +0.05 F1 | Felaktig “ML intent”-positionering |
| Positionering | Svenska + EU-residency + analytics-lager ovanpå valfri CCaaS | Feature-race mot NICE/Genesys förloras |
| Pris (riktning) | Undvik seat-krig; sälj per-analyserad-minut / QA-täckning / residency-SLA | Undervärdera on-prem; jämförs med USD 150–300 AI-tiers |

---

## 6. 90-dagars prioritering

### P0 (vecka 1–4) — gör piloten ärlig

| # | Åtgärd | Ägare / yta |
|---|--------|-------------|
| P0.1 | Leverera DATA-01 anonymiserad korpus + import via `scripts/import_domain_corpus.py` | Data/juridik + `data/import/` |
| P0.2 | Stäng L7–L9 + webui live `/testlab` + spotcheck (PRODUCTION_CHECKLIST) | Ops |
| P0.3 | Lås prod-config: local ASR, Groq off, anonymize on, OpenRouter allow-list | `configs/`, `.env`, `SECURITY.md` |
| P0.4 | DPIA-utkast + leverantörs-DPA (OpenRouter/Mistral; HF endast Hub-download) | Juridik |

### P1 (vecka 5–8) — kvalitet som går att visa

| # | Åtgärd | Ägare / yta |
|---|--------|-------------|
| P1.1 | Telefoni-WER på ≥100–500 riktiga svenska samtal (KB-Whisper) | Eval / `src/evaluate.py`, `scripts/` |
| P1.2 | Sentiment LoRA / domain fine-tune mot gate ≥0.70/0.68 på *riktig* val | `configs/analyzer_eval.yaml`, training extra |
| P1.3 | Intent: träna + `compare_intent_backends.py`; switch endast vid +0.05 | intent train scripts |
| P1.4 | Ersätt Fas 4 KPI-stubs med mätbara definitioner där labels finns | `src/evaluate.py` |

### P2 (vecka 9–12) — skalning & polish

| # | Åtgärd | Ägare / yta |
|---|--------|-------------|
| P2.1 | OTLP production endpoint | `src/core/tracing.py` |
| P2.2 | Redis pub/sub för `TranscriptionEventHub` om multi-worker krävs | `src/api/` |
| P2.3 | Dashboard: executive drill-downs / correlation (ROADMAP medium) | `webui/` |
| P2.4 | Deepgram EU endast efter DPA — som latency-escape hatch, ej default | cloud-stt + SECURITY |

**Utanför 90 dagar:** YouTube ingest, full WS-first rewrite, feature-parity med WFM-sviter.

---

## 7. Riskregister

| Risk | Sannolikhet | Impact | Mitigering |
|------|-------------|--------|------------|
| Överlova accuracy på synthetic baselines | Hög | Hög | Extern gate: ≥500 real calls; ingen kund-WER utan mätning |
| Cloud STT raw audio (Deepgram) | Medel om misconfig | Kritisk | Default local; prod deny-list; metrics `asr_cloud_egress` |
| Groq i prod | Medel om “snabbare” | Hög | Feature-flag + code gate; policy i STRATEGY |
| Article 9 i samtal (hälsa m.m.) | Medel i vissa vertikaler | Kritisk | Local-only path; segmentera kunder |
| Multi-worker WS inkonsistens | Medel vid scale-out | Medel | Single-worker eller vänta på pub/sub |
| CCaaS bundling äter budget | Hög | Medel | Sälj analytics-lager + residency, inte “ersätt Genesys” |
| LLM-kostnadsspridning (A/B compare) | Medel | Låg–medel | Behåll 0.08 budget; mät `llm_requests_total` |

---

## 8. Öppna frågor (kräver mänskligt svar)

1. Primär pilotkund: bank/försäkring, telco, BPO eller intern QA?
2. Förväntad audio-volym (min/mån) — under eller över ~150–220k break-even?
3. Finns Article 9-risk i målkö (hälsa, fack, biometri)?
4. Vem äger annotering (MQM/preference) och budget?
5. Ska deep-path coaching vara premium-tier eller included i pilot?
6. Har ni kapacitet för GPU-host i EU/Sverige för self-host ASR?

---

## 9. Källor

### Internt

- [docs/ROADMAP.md](ROADMAP.md)
- [docs/PRODUCTION_CHECKLIST.md](PRODUCTION_CHECKLIST.md)
- [docs/ANALYZER_STRATEGY.md](ANALYZER_STRATEGY.md)
- [SECURITY.md](../SECURITY.md)
- [reports/domain_baseline.json](../reports/domain_baseline.json)
- [configs/analyzer_eval.yaml](../configs/analyzer_eval.yaml)
- [configs/llm_config.yaml](../configs/llm_config.yaml)
- [docs/plans/2026-07-17-decision-surface-brainstorm.md](plans/2026-07-17-decision-surface-brainstorm.md)

### Parallel deep research

- [nordic-cc-intelligence-market-2026.md](../reports/parallel-research/nordic-cc-intelligence-market-2026.md) (trun_0da5f2efc6af4a8c9217971f61f61b9f)
- [gdpr-asr-llm-eu-residency-2026.md](../reports/parallel-research/gdpr-asr-llm-eu-residency-2026.md) (trun_0da5f2efc6af4a8cb4e80d09734f4231)
- [asr-llm-cost-slo-callcenter-2026.md](../reports/parallel-research/asr-llm-cost-slo-callcenter-2026.md) (trun_0da5f2efc6af4a8cb6fb393a0d70f72e)
- [swedish-asr-sentiment-eval-sota-2026.md](../reports/parallel-research/swedish-asr-sentiment-eval-sota-2026.md) (trun_0da5f2efc6af4a8c9c26763c5f6cbb85)

### Externa URL:er citerade i Parallel-underlag (urval)

- [KBLab — Welcome KB-Whisper](https://kb-labb.github.io/posts/2025-03-07-welcome-KB-Whisper) (Mar 2025)
- [ConnexAI ASR benchmark](https://connex.ai/us/resources/connexai-leading-automatic-speech-recognition-benchmark) (Feb 2026)
- [Deepgram ASR buyer’s guide](https://deepgram.com/learn/asr-buyers-guide-benchmarks-to-production-tests) (2026)

---

*Detta dokument är beslutsunderlag, inte juridisk rådgivning. DPIA/DPA ska granskas av kompetent counsel.*
