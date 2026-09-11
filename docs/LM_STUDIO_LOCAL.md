# LM Studio lokal LLM-integration

Denna dokument beskriver integrationen av lokalt hostade LM Studio-modeller i
sentimentanalys-pipelinen. Målet är lokal-first analys efter transkription utan
cloud-beroende, med Qwen 3.5 9B The Defiant Fable (Q4_K_S) som referensmodell.

## Arkitekturöversikt

```
Ljud → ASR (CPU) → diarization → PII-redaction → lokala analyzers
                                                          ↓
                                               LM Studio (GPU, 127.0.0.1:1234)
                                                   ↓
                                               CallLLMOutput (strukturerad JSON)
                                                   ↓
                                               validering + evidenskontroll
                                                   ↓
                                               QA / rapport / insikter
```

LM Studio körs på samma Windows-maskin som backend. ASR körs explicit på CPU så
att GPU:n (RTX 5070) är reserverad för LM Studio-inferens.

## Konfigurerad modell

| Egenskap | Värde |
|---|---|
| Model key | `qwen3.5-9b-the-defiant-fable-uncensored-heretic-neo-imatrix-max-mtp` |
| Kvantisering | `Q4_K_S` |
| Modellstorlek | ~8.38 GB |
| Max kontext (modellkort) | 262 144 |
| **Målkontext (policy)** | **70 000 total** (instruktioner + utdata) |
| Reasoning default | av (önskat) — se kompatibilitetsavsnitt |
| Samtidighet | 1 aktiv inferens |

## Tokenbudget

```
70 000 total kontext
− 8 192 reserverad utdata
− 2 048 säkerhetsmarginal
= 59 760 max formaterad input
```

Budgeten inkluderar systeminstruktioner, användarprompt, rollmärkt transkription,
lokala analyzer-resultat, JSON-schema-overhead och reasoning-overhead. Överflödig
input avvisas explicit — ingen tyst trunkering av viktig transkription.

Tokenräkning använder LM Studio SDK:ns tokenizer och prompt-mall när tillgängligt
(`LMStudioClient.count_chat_tokens`), inte en generisk approximation.

## Provider-val

LM Studio är en valbar lokal profil (`lmstudio`), inte en global ersättning av
cloud-providers. Befintliga OpenRouter/Mistral/NVIDIA/Cerebras/Groq-profiler
behålls oförändrade.

- CLI: `--provider lmstudio`
- API: `provider: "lmstudio"` i pipeline- eller analysjobb-begäran
- WebUI: lokal profil i Testlabb/Transkriptionssidan

Ingen cloud-API-nyckel krävs för den lokala profilen. En placeholder-nyckel
(`lm-studio`) används internt för OpenAI-kompatibla klienter.

## LLM-vägar som kopplats till gemensam klientpolicy

Alla post-transkriptionella LLM-vägar använder `src/llm/client_factory.py`:

- Holistisk analys (`ConversationMistralAnalyzer` via `pipeline_steps.py`)
- QA/compliance (`src/compliance_qa.py`) — full transkription, inte första 12 segment
- Sentiment-judge (`src/analysis/llm_judge.py`)
- Insights-aggregation (`src/pipeline.py`)
- Jämförelseendpoints där tillämpligt

Lokala deterministiska analyzers förblir basväg. LM Studio-utdata valideras och
kan inte tyst ersätta lokala resultat utan provenance.

## Reasoning-off kompatibilitet (viktig begränsning)

**Reasoning av kan inte garanteras via LM Studios OpenAI-kompatibla API.**

Live-prov (2026-09) med Qwen 3.5 9B The Defiant vid 70 144 aktiv kontext:

| Parameter | Värde |
|---|---|
| `response_format` | `json_schema` (strict) |
| `finish_reason` | `stop` |
| `content` (slutgiltig JSON) | giltig, 120 tecken |
| `reasoning` | 6 138 tecken, 1 612 reasoning_tokens |
| Pydantic-validering | OK |

Modellen emitterar alltså reasoning-tokens även med strukturerad utdata. Detta
innebär:

1. **Tokenbudgeten måste räkna med reasoning-overhead.** Säkerhetsmarginalen
   (2 048) är vald med detta i åtanke, men vid gränsfall kan reasoning konsumera
   utdata-budgeten.
2. **Slutgiltig `content` är giltig JSON** och validerar mot `CallLLMOutput`.
   Klienten accepterar endast `content` som svar — reasoning ignoreras och loggas
   aldrig i klartext.
3. **`enable_thinking=false` och `chat_template_kwargs` ignoreras** i nuvarande
   LM Studio-version för denna modell. Detta är en känd LM Studio/Qwen 3.5
   kompatibilitetsfråga, inte en applikationsbugg.

Applikationen hanterar detta genom att:

- Kräva icke-tom `content` (reasoning-only svar avvisas).
- Validera `content` mot `CallLLMOutput`-schemat.
- Avvisa trunkerad/ogiltig JSON.
- Degenerera till lokala analyzer-resultat om validering misslyckas.

## Felhantering och graceful degradation

Om LM Studio inte är nåbar, modellen inte är laddad, kontexten är otillräcklig
eller validering misslyckas:

- **Ingen automatisk cloud fallback.**
- Pipelinen degenererar till lokala deterministiska analyzers.
- `degraded`-lista och `mode: "degraded"` markeras i rapporten.
- QA behåller regelbaserad fallback.

Preflight (`src/install/preflight.py`) rapporterar:

- LM Studio-reachability på loopback
- Laddad modell och kvantisering
- Aktiv kontext vs policy
- Reasoning-default

## Bakgrundsjobb för långa analyser

Synkron `/analyze_pipeline` behåller 200-segmentsgränsen för kompatibilitet.
För längre transkriptioner används beständiga bakgrundsjobb:

| Endpoint | Syfte |
|---|---|
| `POST /analysis/jobs` | Köa analys (HTTP 202 + job_id) |
| `GET /analysis/jobs/{id}` | Status + fas |
| `GET /analysis/jobs/{id}/result` | Slutgiltig rapport |
| `POST /analysis/jobs/{id}/cancel` | Avbryt begäran |

Jobbegenskaper:

- Payloadgräns 2 MiB, segmentgräns dokumenterad i API-schemat
- SQLite-backad lagring av status/resultat (ingen råljud)
- Idempotens-nyckel via `Idempotency-Key`-header
- Restart-återhämtning: körande jobb markeras `interrupted`, inte `completed`
- Sena resultat från avbrutna jobb kasseras
- Max 1 samtidig lokal inferens

Tillstånd: `queued`, `running`, `cancel_requested`, `cancelled`, `completed`,
`completed_degraded`, `failed`, `interrupted`.

Faser: `local_analysis`, `llm_holistic`, `qa`, `validating`, `persisting`.

## PII och säkerhet

- Analysis jobs tvingar PII-redaction (`force=True`) oavsett profil innan persist.
- Oredigerad PII skickas inte till LM Studio när `anonymize_before_llm` är satt (alla `llm.enabled`-profiler) eller när jobs `force=True`. Profiler utan flaggan no-opar fortfarande i den vanliga pipeline-vägen.
- Om tidig PII-redaction misslyckas och profilen kräver anonymisering, hoppas
  all LLM-anrikning över.
- Prompts, transkriptioner, reasoning-innehåll och råa svar loggas aldrig.
- Lokala anrop markeras med `LOCAL LLM CALL`-loggningskonvention.

## Pilotpolicy

Pilot-runbook (`docs/PILOT_RUNBOOK.md`) auktoriserar för närvarande endast
OpenRouter → Mistral EU/ZDR och lokal ASR. **LM Studio för kunddata kräver en
separat policybeslut** innan pilotbruk. Denna integration är tekniskt redo men
inte automatiskt pilot-godkänd.

## Manuell LM Studio-konfiguration

Applikationen ändrar aldrig LM Studios modellinställningar automatiskt. För att
nå 70k kontext:

1. Öppna LM Studio → ladda modellen
2. Sätt context length till ≥ 70 000 (t.ex. 70 144)
3. Sätt antal instanser till 1 (max samtidighet = 1)
4. Flash Attention på, MTP av (valfritt)
5. Starta servern på `127.0.0.1:1234`

Verifiera med:

```bash
python scripts/probe_lmstudio_live.py
```

## Filer

| Fil | Syfte |
|---|---|
| `src/llm/lmstudio_client.py` | OpenAI-kompatibel klient + native status |
| `src/llm/client_factory.py` | Delad provider/client-konstruktion |
| `src/llm/context_budget.py` | 70k tokenbudget |
| `src/llm/response_validation.py` | Svars-/evidensvalidering |
| `src/api/services/analysis_jobs.py` | Bakgrundsjobb-hantering |
| `configs/llm_providers.yaml` | LM Studio-providerkonfiguration |
| `scripts/probe_lmstudio_live.py` | Live kompatibilitetsprov |
| `tests/test_lmstudio_client.py` | Klient-/status-/budgettester |
| `tests/test_llm_response_validation.py` | Valideringstester |
| `tests/test_analysis_jobs.py` | Jobblivscykel-tester |
