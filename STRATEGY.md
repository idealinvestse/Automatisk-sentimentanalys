---
name: Automatisk-sentimentanalys
last_updated: 2026-07-17
---

# Automatisk-sentimentanalys Strategy

## Target problem

Nordiska kontaktcenter och QA-team behöver förstå svenska kundsamtal i stor skala, men internationella speech-analytics-sviter är dyra att byta till, svaga på svensk telefoni och tunga kring EU-dataresidency. Utan lokal-först analys tvingas känsliga samtal ut till moln-STT/LLM — eller så blir kvalitetssäkring stickprovsbaserad och långsam.

## Our approach

Bygg ett hybrid **local-first** Call Center Intelligence-lager: svensk ASR (KB-Whisper) och lokala analyzers som default, selektiv EU-alignad LLM endast för högvärdesresonemang efter PII-redaction. Konkurrera på språk + residency + ärlig degradation — inte på att ersätta hela CCaaS/WFM-sviter.

## Who it's for

**Primary:** QA- och contact-center-ledare i reglerade nordiska organisationer (bank, försäkring, telco, offentlig sektor) - They're hiring Automatisk-sentimentanalys to cover fler samtal med svensk kvalitet och EU-kontrollerad datahantering utan att rip-and-replace Genesys/NICE.

## Key metrics

- **Telephony WER (svenska)** - Word error rate på representativ call-center-ljudslice; mäts offline i eval, inte på FLEURS
- **Sentiment accuracy / macro-F1** - På *riktig* domain-val efter DATA-01; CI-trösklar i `configs/analyzer_eval.yaml`
- **Intent macro-F1** - Heuristic tills modell slår +0.05; mäts på `data/intent_val.jsonl` + framtida domain-set
- **LLM cost per analyzed call** - Soft budget (default USD 0.08); mäts via pipeline/metrics
- **Cloud egress incidents** - Antal opt-in cloud-STT/LLM-anrop med PII-risk (ska vara noll i default-pilot)

## Tracks

### Swedish speech quality

Investera i lokal ASR, diarization och domain-eval på verklig telefoni så att svenska blir bevisbar fördel.

_Why it serves the approach:_ Utan mätbar svensk WER kollapsar differentieringen till generisk feature-lista.

### Privacy-preserving intelligence

Håll PII-redaction, local-default och selektiv EU-LLM som produktlöfte; behandla Groq/Deepgram som undantag med grindar.

_Why it serves the approach:_ Residency och Article 9-risk är köpskäl och säljblockerare samtidigt.

### Call-center decisioning

Agent performance, QA scorecards, insights och alerting ovanpå transcript — analytics-lager som fungerar bredvid befintlig CCaaS.

_Why it serves the approach:_ Köparen betalar för beslut (QA, coaching, hot topics), inte för rå transcript.

## Milestones

- **2026-Q3** - Conditional kundpilot under local-ASR + DPIA + DATA-01 minikorpus
- **2026-Q4** - Domain-validerade sentiment/intent-gates på riktig korpus; intent-modell endast om +0.05 F1

## Not working on

- YouTube / multichannel ingest (Fas 5) som kärnprodukt
- Full WS-first realtidsprodukt-rewrite i närtid
- Feature-parity-race mot NICE/Genesys WFM-sviter
- Groq eller cloud-STT som production default för PII-samtal

## Marketing

**One-liner:** Svensk call center-intelligence — lokal ASR först, EU-säker LLM när det lönar sig.

**Key message:** Vi ersätter inte er contact-center-plattform. Vi ger er svensk kvalitet, fullare QA-täckning och data som stannar där ni bestämmer — med ärliga gränser när molnet inte är lämpligt.
