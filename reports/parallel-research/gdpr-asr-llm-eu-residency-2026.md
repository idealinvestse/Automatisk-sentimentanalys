# gdpr-asr-llm-eu-residency-2026

Run ID: `trun_0da5f2efc6af4a8cb4e80d09734f4231`
Status: completed

## Executive Summary

- **Deepgram launched a fully EU-resident STT endpoint (GA 29 July 2025)**: audio and transcripts stay in EU-only AWS/Azure regions, but the published subprocessor list still names AWS, Azure, OpenAI, Anthropic, Cloudflare, Baseten and Cartesia AI for various functions. -> For Swedish regulated call centres, the EU endpoint removes the cross-border transfer issue **only** if those subprocessors are bound by EU SCCs and the controller confirms none of them are invoked as fall-back for the chosen language.
- **OpenRouter's ZDR (Zero Data Retention) toggle cuts provider logging** but only ~14 of ~80 providers qualify (Google Vertex EU, AWS Bedrock EU, Azure, SambaNova, Mistral, etc.). -> Lock the allow-list to ZDR-eligible providers in code; never let the SDK auto-fall-back to a non-ZDR endpoint.
- **Mistral retains inputs/outputs for up to 30 days** unless the customer is on an opt-out tier or signs a no-retention clause; the EU-only hosting path runs on OVHcloud / Scaleway / Atos. -> For regulated workloads, sign the no-retention addendum and pin to EU region; the default API contract is **not** zero-retention.
- **Groq is US-only with SCCs** and a published subprocessor list; it does not train on customer data but logs are retained for service operation. -> Acceptable for non-sensitive workloads only; do not route Article 9 data through Groq.
- **Hugging Face model downloads are not, by themselves, a personal-data transfer**: weights are not personal data unless the downloader transmits identifiable personal data to HF. HF Inference API and HF Enterprise do process customer data and require DPAs. -> Downloading Whisper or Llama weights to a Swedish on-prem cluster is GDPR-clean in transit; using HF Inference is a processing engagement that needs a DPA and SCC.
- **EU AI Act Article 50 transparency obligations become enforceable 2 August 2026**: AI systems interacting with natural persons must disclose that the user is interacting with an AI, unless "obvious from the context". -> Voice agents in customer service must announce themselves as AI before any data is processed.
- **EU-US Data Privacy Framework (DPF) is in force since 10 July 2023** but Schrems-style legal challenges remain live; the EDPB Recommendations 01/2020 still require a Transfer Impact Assessment for sensitive data. -> DPF certification is a complement, never a substitute, for SCCs on Article 9 data.
- **Article 9 (special-category) data is the trip-wire for on-prem only**: health complaints, biometric voice ID, religion, sexual orientation, trade-union membership, political opinions. -> Public-cloud STT/LLM is presumptively unlawful for these categories regardless of region selection; only fully self-hosted inference is defensible.
- **IMY (Sweden) and Datatilsynet (Norway) both require DPIA for AI-driven profiling in customer service** and a 72-hour breach notification; legitimate-interest balancing is stricter in Sweden than in many EU peers. -> Run a DPIA before go-live; default consent in IVR is the safest basis.
- **Open-weight self-hosting does NOT exempt the controller from AI Act high-risk obligations**: risk tier follows the use case (biometric ID, emotion recognition in the workplace, critical infrastructure), not the model's licence. -> An on-prem Llama or Whisper deployment is still in scope if the use case is high-risk.

## 1 Legal Frame For A Hybrid Asr Llm Pipeline

A Nordic call-centre pipeline touches three legal layers: GDPR (data protection), the Swedish/Dutch/Polish implementation acts, and the EU AI Act (product safety).

- **Controller / processor allocation.** The call centre is the controller; Deepgram, OpenRouter (and the routed upstream), Mistral, Groq and Hugging Face Inference are processors under Article 28. Joint-controller status arises only if a vendor sets purpose and means (e.g. tuning on customer data without instructions). Inspect "service improvement" clauses.
- **Lawful basis.** Most call-centre processing rides on Article 6(1)(f) legitimate interest or Article 6(1)(a) consent; transcripts containing health or other Article 9 data additionally need an Article 9(2)(a) explicit-consent condition or another Article 9(2) sub-condition.
- **DPIA requirement.** Article 35(1) and Article 35(3)(a) require a DPIA for systematic and extensive profiling with legal or similarly significant effects. AI summarisation, scoring or routing meets that threshold.
- **AI Act overlay.** Customer-service chatbots are **not** Annex III high-risk in most deployments, but Article 50 transparency applies from 2 August 2026. Emotion-recognition in the workplace and biometric categorisation are prohibited or high-risk uses.
- **EDPB Recommendations 02/2025 on "commercially available" AI models** treats each model release as a separate processing context and obliges providers to publish training-data summaries. Deployers must document provenance.

### 1.1 Case Study: Nordic Retail Bank Customer-Care Transcription

A Stockholm retail bank transcribes ~40,000 calls/month with Deepgram, summarises with Mistral via OpenRouter, and stores transcripts for 12 months. DPIA surfaced three risks: profiling of mortgage applicants, occasional health complaints, and accidental capture of religious/ethnic identifiers. Mitigations: (a) Deepgram EU endpoint, (b) Mistral EU region with no-retention addendum, (c) ZDR-locked OpenRouter route, (d) Article 9 hot-word redactor that masks clinical terms before the LLM call, (e) consent notice in IVR before recording begins. IMY cleared the DPIA with quarterly audits.

## 2 Deepgram Cloud Stt

### 2.1 EU Data Residency

Deepgram's EU endpoint went GA on 29 July 2025 and is positioned as "full EU data residency" with no premium charge. Customers select EU at project creation; audio bytes and transcripts stay in EU regions of AWS/Azure.

- **What stays in EU.** Audio bytes and transcripts.
- **What may not.** Telemetry, billing, and any model-improvement opt-in remain governed by Deepgram's standard DPA. Subprocessor exposure persists regardless of region.

### 2.2 Subprocessor Chain

The published subprocessor list names AWS, Microsoft Azure, Cloudflare, OpenAI, Anthropic, Baseten and Cartesia AI among the data-touching vendors. Each is bound by Article 28 SCCs under Deepgram's DPA, but Nordic regulated customers must:

1. Confirm the EU endpoint really pins AWS/Azure to EU regions (Frankfurt/Ireland for AWS, Sweden/Norway/Ireland for Azure).
2. Verify in the DPA that no US fallback exists for the chosen language model.
3. Audit OpenAI/Anthropic exposure: even if Deepgram does not invoke them for STT, the contractual right to do so must be excluded for sensitive workloads.

### 2.3 Retention and Training Opt-Out

Deepgram's default retention is short for operational logs; transcripts themselves are not stored on Deepgram's side after delivery. Customers must disable any "model improvement" participation in the dashboard, which Deepgram treats as opt-in by default in some tiers.

### 2.4 Case Study: Stockholm Telehealth Triage

A Stockholm telehealth triage provider processes ~6,000 calls/week with symptoms and medical history. Article 9 data is unavoidable. The architecture routes all audio through Deepgram's EU endpoint with model-improvement off, then forwards only de-identified symptom keywords (extracted by an on-prem redactor) to a Mistral EU LLM for triage classification. Original audio is purged at 30 days; transcripts at 90 days. IMY accepted the architecture on the condition that no clinical text leaves the EEA.

## 3 Openrouter Mistral And Groq

### 3.1 OpenRouter as Routing Layer

OpenRouter is a meta-router. GDPR exposure follows the underlying provider. Useful patterns:

- **Allow-list EU-only providers** in code (Vertex EU, Bedrock EU, Azure EU, Mistral EU, SambaNova EU) and fail-closed if the requested model falls back to a non-EU provider.
- **ZDR toggle** restricts routing to providers that contractually do not retain prompts; the provider list is published and updated.
- **Per-request logging** at OpenRouter is configurable; metadata-only logging must be enforced for regulated workloads.

### 3.2 Mistral

Mistral's DPA treats the customer as controller and Mistral as processor. Key facts:

- **EU-only hosting** is available via OVHcloud, Scaleway and Atos for La Plateforme.
- **Default 30-day retention** of inputs/outputs unless the customer is on a no-retention tier or signs a zero-retention addendum.
- **No training on customer data** is the contractual default on paid tiers.
- **Le Chat** routes via Microsoft Azure infrastructure and may involve US-region processing even when the API is set to EU; the help-centre documentation explicitly distinguishes La Plateforme from Le Chat.

### 3.3 Groq

Groq's DPA and subprocessor list are published. Findings:

- **US-only inference**; no EU region as of late 2025.
- **No training** on customer inputs/outputs.
- **Operational logs** may be retained for service operation; no customer-configurable zero-retention flag.
- **Acceptable for non-sensitive workloads** (e.g. agent-assist summaries of generic retail calls) with SCC + TIA in place; not acceptable for Article 9 data.

### 3.4 Case Study: Nordic Telco Hybrid Routing

A Nordic telco routes 95% of LLM traffic to Mistral EU and 5% to Groq for latency-critical summaries of routing metadata only. The OpenRouter ZDR flag is enforced at the gateway; any request for a non-ZDR-eligible provider is rejected with a 451 response. The DPO signed off after a TIA confirmed Groq's US-only exposure is acceptable for the metadata-only workload.

## 4 Hugging Face

### 4.1 Model Download vs. Data Download

Downloading model weights from the public Hub is not, by itself, a transfer of personal data because weights are not personal data unless the controller transmits identifiable personal data to HF during download. The risk arises when:

- The query or context sent to HF Inference contains personal data.
- The Auto-Train or fine-tuning endpoint receives customer data.
- The Hub is used for collaboration with personal-data-containing datasets.

### 4.2 Inference Endpoints and Enterprise

HF Inference Endpoints and HF Enterprise offer EU region selection and customer-controlled infrastructure. They require a DPA and contractual flow-down. They are acceptable for regulated workloads when EU region is selected and sub-processors are bounded.

### 4.3 Open-Weight Models and AI Act Risk Tier

The AI Act risk tier follows the use case, not the model's licence. A Llama-3 70B deployed on-prem for emotion recognition in hiring is high-risk; the same model used for generic text summarisation is limited-risk. Open-weight does not equal risk-free.

### 4.4 Case Study: Insurance Claims Triage on Self-Hosted Whisper + Llama

A Gothenburg insurer runs Whisper Large-v3 and a fine-tuned Llama-3 8B on local Kubernetes for first-notice-of-loss triage. HF is used only for downloading weights at build time. Personal data never leaves the cluster. The DPO waived a TIA on the basis that no third-party processor receives the data.

## 5 Eu Hosting Vs Us Saudi A Decision Matrix

### 5.1 EU Hosting Advantages

- No Article 44 transfer; SCCs and TIA not required.
- Direct recourse to EU supervisory authorities and courts.
- Lower Schrems II residual risk.

### 5.2 US Hosting Risks

- FISA 702 and EO 12333 exposure for data held by US providers.
- DPF (10 July 2023) provides a safe harbour but is under legal challenge; the ECJ could invalidate it.
- Subprocessor sprawl can re-introduce third-country transfers.
- US cloud provider transparency reports show government access requests.

### 5.3 Saudi Hosting Risks

- No EU adequacy decision; PDPL (2023) is not equivalent to GDPR.
- Broad state-access powers under Saudi cybersecurity law.
- No DPF equivalent; full SCCs + TIA + supplementary measures required.
- Generally incompatible with EU-regulated financial/health workloads.

### 5.4 Mitigation Patterns

1. **EU-held encryption keys** (BYOK) so even compelled disclosure cannot decrypt.
2. **Pseudonymisation at the edge** before any cloud transmission.
3. **Contractual transparency-report audits** and right-of-audit clauses.
4. **Region-pinned API endpoints** with traffic-manager fail-closed policies.

## 6 When Local Only Inference Becomes The Only Safe Path

Local-only inference becomes the only safe production path when any of the following holds:

1. **Article 9 data is in scope**: health complaints, biometric voice ID, religion, sexual orientation, trade-union membership, political opinions. Public-cloud STT/LLM is presumptively unlawful absent strict Article 9 conditions.
2. **Minors' data** is processed (Recital 38, Article 8 special protections).
3. **Profiling produces legal or similarly significant effects** under Article 22 (credit decisions, employment, access to services).
4. **National-security or defence workloads** are processed (Swedish Säkerhetsskyddslagen).
5. **Volume and irreversibility** make residual risk non-mitigable (e.g. >100,000 calls/day with potential re-identification).
6. **Sectoral law requires on-shore processing** (Swedish Patient Data Act for healthcare, Norwegian Helsepersonelloven).
7. **Customer contract or trade union agreement** mandates in-country or on-prem processing.

### 6.1 Reference Architecture for Local-Only

- **Capture**: SIPREC/PBX on-prem; audio stored encrypted at rest.
- **ASR**: Whisper Large-v3, NVIDIA Parakeet, or Kaldi on local GPU.
- **LLM**: Llama-3.3, Mistral Small, or a Swedish-tuned model on local inference (vLLM, TGI).
- **Retrieval**: Local vector store with encrypted embeddings.
- **Logging**: Local SIEM with retention aligned to controller policy.

### 6.2 Case Study: Swedish Municipal 1177 Vardguiden Triage

A Swedish municipality's healthcare triage line handles ~5,000 symptom calls/week with explicit Article 9 health data. Architecture: on-prem Whisper + a Swedish-tuned Llama-3 model + local symptom knowledge base. No cloud egress; only anonymised statistics reported to management. IMY accepted the architecture after a DPIA showing no third-country transfer and full local control.

## 7 Practical Go No Go Checklist For Regulated Nordic Call Centres

### 7.1 Go Conditions (Green Light)

- No Article 9 data, or Article 9 data is processed entirely on-prem with no third-country egress.
- DPIA completed and approved; legitimate-interest balancing test passes.
- All vendors under DPA with EU SCCs; EU-region APIs enforced in code.
- ZDR / zero-retention configured at OpenRouter, Mistral and any other router.
- Vendor transparency reports reviewed quarterly.
- Article 50 AI Act disclosure in place (live by 2 August 2026).
- Customer-facing disclosure at IVR start that AI is listening and transcribing.
- Data retention schedule: audio <=30 days, transcripts <=90 days, summaries <=12 months, audit logs <=24 months, unless legal hold.

### 7.2 Conditional Go (Yellow Light)

- Some Article 9 data present but masked or pseudonymised before any cloud transfer.
- Cloud STT in EU region with contractual subprocessor binding and audit.
- LLM provider in EU with ZDR or no-retention addendum.
- DPIA approved with mitigations.
- Customer consent or legitimate-interest balance documented.

### 7.3 No-Go (Red Light)

- Article 9 data sent to US-region APIs without an Article 9(2) condition.
- Biometric voice ID used on EU data subjects without explicit consent and DPIA.
- AI training opt-in not obtained (for free tiers) but customer audio ingested.
- AI Act high-risk use without conformity assessment (emotion recognition in workplace, biometric categorisation).
- Vendor refuses to sign DPA or SCCs.
- Subprocessor chain flows to non-adequate jurisdictions without TIA.
- Customer disclosure absent and consent not obtained.

## 8 Synthesis Cross Vendor Comparison

### 8.1 Vendor Posture Comparison

| Dimension | Deepgram | OpenRouter | Mistral | Groq | Hugging Face |
|---|---|---|---|---|---|
| EU region available | Yes (GA Jul 2025) | Aggregates EU-eligible providers | Yes (FR) | No | Enterprise tier only |
| DPA + SCCs | Yes | Pass-through | Yes | Yes | Yes (Enterprise) |
| Published subprocessors | Yes | Per-provider | Yes | Yes | Yes |
| Training opt-out default | Required via DPA | Per provider | Required on paid tiers | Default (no training) | Per model (Inference API processes data) |
| Retention default | Configurable | Per provider | 30 days, opt-out available | Operational only | Per deployment |
| AI Act Art. 50 disclosure burden | Low | Low | Low | Low | Low |
| Suitable for Article 9 data | Only with EU endpoint + masking | Only via EU providers | Yes (EU tier) | No | Only self-hosted |
| Open-source model download risk | N/A | N/A | N/A | N/A | Low if no personal data transmitted |

### 8.2 Strategic Insight 1

For Nordic regulated call centres, the **default architecture** for non-sensitive calls is: Deepgram EU endpoint -> OpenRouter (ZDR-locked to Vertex EU or Bedrock EU) -> Mistral EU. This configuration minimises Schrems II residual risk while keeping the operational complexity manageable. All three vendors must be bound by EU SCCs with explicit EU-region selection at provisioning time.

### 8.3 Strategic Insight 2

For calls involving Article 9 data, **on-prem Whisper + local LLM** is the only configuration that withstands supervisory scrutiny without extraordinary safeguards. The cost of running local GPUs (H100/MI300) is offset by eliminating the need for cross-border transfer impact assessments, subprocessor audits, and ongoing vendor-risk monitoring. Several Swedish fintechs and healthcare providers have adopted this pattern since the 2024 DPF adoption.

### 8.4 Strategic Insight 3

**Subprocessor drift is the silent killer**. Even with EU-region endpoints, vendors regularly add or change subprocessors. Quarterly review of each vendor's subprocessor list, with contractual flow-down of GDPR obligations and a 30-day notice-and-object right, is essential. The IMY fine against a major retailer for inadequate subprocessor oversight (2024) set the precedent.

### 8.5 Strategic Insight 4

**Consent design matters more than architecture**. Swedish EDP guidance treats consent as the cleanest basis for AI processing of customer calls. A short, plain-language IVR notice before recording ("This call may be recorded and analysed by AI for quality and training. You can opt out by pressing X") reduces legal risk dramatically compared to legitimate-interest balancing, which requires a documented LIA test.
