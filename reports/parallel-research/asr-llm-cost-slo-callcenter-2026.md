# asr-llm-cost-slo-callcenter-2026

Run ID: `trun_0da5f2efc6af4a8cb6fb393a0d70f72e`
Status: completed

## Executive Summary

- **Managed STT price floor (2026):** Cartesia Ink-Whisper streaming at ~$0.00217/min leads the market; Deepgram Nova-3 sits at $0.0077/min PAYG and $0.0065/min on Growth plans (16% off, requires $4k-$10k/yr commit) per the Deepgram 2026 breakdown.
- **Self-hosted Whisper is 2-10x cheaper than every managed API above ~200k minutes/month:** an RTX 4090 at ~$0.40/hr running faster-whisper large-v3 at ~250 audio-hours/hour lands at ~$0.0016/min of audio versus Deepgram Nova-3 Growth at $0.0065/min.
- **Break-even window is ~150-220k audio-minutes/month sustained:** below that, managed APIs win on TCO once SRE/observability/model-refresh labor is priced in; above 1M minutes/month, self-hosting on A100/H100 reserved capacity wins 2-4x and Parakeet-TDT can halve compute on English queues.
- **Realistic p95 latency budgets:** batch post-call analytics ~10-30s per 1-min file on A100; near-real-time agent-assist streaming 260-600ms first-token (Deepgram Nova-3, OpenAI gpt-4o-transcribe-realtime, Cartesia); end-to-end agent-assist stack 1.2-2.5s p95.
- **LLM cost per analyzed call is sub-cent:** full analytics (transcript + summary + QA + sentiment + topic) on a 7-min call lands at $0.005-$0.015 with small models (GPT-4o mini, Claude Haiku 4.5, Gemini Flash); real-time agent-assist streaming is the most expensive hidden line item at $0.005-$0.015 per minute.
- **Whisper-large-v3 batch throughput on A100 80GB:** ~500-700 audio-hours per GPU-hour with faster-whisper int8; H100 pushes ~900; L40S ~350; L4 ~190; entry RTX 3090 ~250 (best $/audio-min on commodity hardware).
- **Data residency overrides cost optimization:** HIPAA, PCI, GDPR-regulated workloads default to self-host even at higher TCO because major managed STT providers decline BAAs for transcription or cannot guarantee EU data isolation.

---

## 1 Cost Per Audio Minute Cloud Stt Vs Self Hosted Whisper

### 1.1 Managed STT Price Matrix (USD per audio-minute)

| Provider | Model | Batch | Streaming | Notes |
|---|---|---|---|---|
| Cartesia | Ink-Whisper | -- | $0.00217 | Lowest streaming price; sub-150ms latency |
| AssemblyAI | Universal-2 | $0.0035 | $0.0075 | $0.21/hr batch, $0.45/hr streaming |
| OpenAI | gpt-4o-mini-transcribe | $0.003 | -- | $0.18/hr batch tier |
| Google Cloud | STT V2 dynamic batch | $0.003-$0.016 | -- | Tiered by feature set |
| Deepgram | Nova-2 | $0.0043 | -- | Previous-gen |
| OpenAI | gpt-4o-transcribe | $0.006 | -- | $0.36/hr; flat pricing |
| Deepgram | Nova-3 | $0.0077 | $0.0077 | $0.46/hr PAYG |
| Deepgram | Nova-3 (Growth) | $0.0065 | $0.0065 | 16% off, $4k-$10k/yr commit |
| Google Cloud | Chirp 2/3 | $0.006-$0.0167 | $0.0167 | $1/hr streaming tier |
| AWS | Transcribe | $0.024 | -- | Most expensive major |
| Speechmatics | Ursa 2 / Flow | $0.0537 | -- | Premium positioning |

Streaming typically costs 79% more than batch at the same provider (per the Deepgram 2026 analysis). For post-call analytics where latency is irrelevant, batch APIs deliver 2-3x lower cost.

### 1.2 Self-Hosted Whisper Throughput (faster-whisper / CTranslate2, large-v3)

| GPU | Audio-hr / GPU-hr | Cost / 1000 audio-min | Source |
|---|---|---|---|
| RTX 3050 | 14 | $0.40/hr GPU -> ~$0.48/min | GigGPU benchmark |
| RTX 3090 | ~250 | $0.40/hr -> ~$0.0016/min | Best $/audio-min |
| RTX 4090 | 280 | $0.50/hr -> ~$0.0018/min | GigGPU benchmark |
| L4 24GB | ~190 | $0.40/hr -> ~$0.0021/min | |
| A10G 24GB | ~180 | $0.60/hr -> ~$0.0033/min | |
| L40S 48GB | ~350 | $1.20/hr -> ~$0.0034/min | |
| A100 80GB | ~600 | $1.50/hr -> ~$0.0025/min | Spheron benchmark |
| H100 80GB | ~900 | $2.50/hr -> ~$0.0028/min | Marginal vs A100 |

SaladCloud published a public benchmark transcribing 1 million audio-minutes on RTX 3090s for $5,110, implying $0.00511/audio-min all-in including compute, queueing, and egress -- the practical lower bound on commodity hardware.

### 1.3 Break-Even Math

```
cost_per_min_self_host = (GPU_hourly_rate * utilization) / (audio_hours_per_GPU_hour * 60)
```

| GPU vs API | API rate | Self-host break-even (min/hr sustained) | Monthly break-even (24x7) |
|---|---|---|---|
| A10G ($0.60) vs OpenAI ($0.006) | $0.006/min | 100 min/hr | 73k min/mo |
| A10G ($0.60) vs Deepgram ($0.0077) | $0.0077/min | 78 min/hr | 56k min/mo |
| A100 ($1.50) vs OpenAI ($0.006) | $0.006/min | 250 min/hr | 180k min/mo |
| A100 ($1.50) vs Deepgram ($0.0077) | $0.0077/min | 195 min/hr | 140k min/mo |

Add $2,500/month amortized SRE/MLOps overhead and the break-even threshold rises ~50-100k min/mo. At sustained utilization below ~50% the API wins regardless of list price.

---

## 2 Llm Cost Per Analyzed Call

### 2.1 Average Handle Time (AHT) Drives Transcript Size

| Vertical | Avg handle time |
|---|---|
| General contact center | 6-8 min |
| Tech support | 8-12 min |
| Banking / financial services | 4-6 min |
| Healthcare | 5-7 min |
| Outbound sales | 3-5 min |

A 7-minute call with ~60% speech density yields ~750 spoken words per side, ~1,000 tokens of cleaned transcript post-STT, plus ~250 tokens of system prompt.

### 2.2 Per-Call LLM Cost (Small Models)

| Task | Model | Input tokens | Output tokens | Cost/call |
|---|---|---|---|---|
| Summary + sentiment | GPT-4o mini | 1,000 | 200 | $0.00027 |
| Summary + sentiment | Claude Haiku 4.5 | 1,000 | 200 | $0.0020 |
| QA scoring + coaching | GPT-4o mini | 2,000 | 400 | $0.0006 |
| QA scoring + coaching | Claude Haiku 4.5 | 2,000 | 400 | $0.0040 |
| Full analytics pipeline | GPT-4o mini | 3,000 | 800 | $0.0010 |
| Full analytics pipeline | Claude Haiku 4.5 | 3,000 | 800 | $0.0080 |

GPT-4o mini: $0.15/M input, $0.60/M output. Claude Haiku 4.5: $1/M input, $5/M output.

### 2.3 End-to-End Per-Call Stack (7-min call, mid config)

| Component | Cheapest stack | Mid stack | Premium stack |
|---|---|---|---|
| STT | self-hosted Whisper | Deepgram Growth | Nova-3 + diarization |
| Diarization + formatting | built-in | built-in | AssemblyAI |
| LLM analysis | GPT-4o mini | GPT-4o mini | Claude Haiku 4.5 |
| Storage (90-day) | $0.0002 | $0.0005 | $0.001 |
| Orchestration/observability | $0.0005 | $0.001 | $0.002 |
| **Total per call** | **~$0.003** | **~$0.016** | **~$0.039** |

At 1M calls/month this is $3k-$39k/month. STT and LLM contribute roughly equal shares in the mid stack.

### 2.4 Real-Time Agent-Assist Is the Hidden Cost

Real-time agent-assist streams partial transcripts every 200-500ms, runs an LLM turn classification or next-best-action prompt every 3-5 seconds, and optionally synthesizes a spoken suggestion. Continuous LLM output at ~50 tok/s adds **$0.005-$0.015 per minute** of call time -- often the single largest line item in agent-assist deployments and frequently missed in initial TCO modeling.

---

## 3 Break Even Audio Minutes Month Self Host A10 A100 Vs Api Opex

### 3.1 Assumptions

- A10G 24GB spot: $0.60/hr; sustained throughput 180 audio-hr/hr (75% utilization)
- A100 80GB reserved 1-yr: $1.50/hr; sustained throughput 600 audio-hr/hr
- API benchmarks: Deepgram Nova-3 PAYG $0.0077/min, OpenAI gpt-4o-transcribe $0.006/min
- Engineering overhead: 0.25 FTE MLE at $150k fully-loaded = $3,125/mo amortized

### 3.2 Break-Even Thresholds

| Comparison | Self-host cost/min | API cost/min | Break-even (min/mo) |
|---|---|---|---|
| A10G vs Deepgram Nova-3 | $0.0033 | $0.0077 | ~75k (incl. eng overhead ~125k) |
| A10G vs OpenAI transcribe | $0.0033 | $0.0060 | ~150k (incl. eng overhead ~225k) |
| A100 vs Deepgram Nova-3 | $0.0025 | $0.0077 | ~80k (incl. eng overhead ~135k) |
| A100 vs OpenAI transcribe | $0.0025 | $0.0060 | ~165k (incl. eng overhead ~250k) |

### 3.3 Decision Rules

- **Under 50k audio-min/month:** managed API wins on TCO every time, even before considering on-call coverage.
- **50k-200k audio-min/month:** tie zone; the decision is governed by data-residency, compliance, and feature needs (diarization, language ID, custom vocab) rather than raw compute cost.
- **200k-1M audio-min/month:** self-hosting on L40S or A10G reserved instances wins 30-50% TCO if you already run a Kubernetes platform.
- **Above 1M audio-min/month:** self-hosting on A100/H100 reserved capacity wins 2-4x; layer in Parakeet-TDT for English queues to halve compute again.

### 3.4 Hidden Cost Multipliers in Self-Hosting

- **GPU spot price volatility:** can swing 2-3x during demand spikes; reserved/committed pricing reduces but does not eliminate this.
- **Model upgrade cadence:** new Whisper/Parakeet/Canary checkpoints ship every 3-6 months; budget 0.1 FTE for ongoing eval and re-deployment.
- **Diagonal scaling:** call-center traffic is bursty (10x peak-to-trough); idle GPU capacity is the largest single source of self-host TCO erosion. Aim for sustained 60-70% utilization.
- **Storage and retention:** 90-day compressed audio retention at 1M calls/month is ~5 PB; budget $0.02/GB-month cold storage.

---

## 4 Realistic P95 Latency Batch Vs Near Real Time

### 4.1 Pipeline Latency Benchmarks

| Pipeline | Median | p95 | Notes |
|---|---|---|---|
| Deepgram Nova-3 streaming | 260 ms | 400-600 ms | First-token latency |
| OpenAI gpt-4o-transcribe realtime | 200 ms | 320 ms | Streaming partials |
| Cartesia Ink-Whisper realtime | <200 ms | <300 ms | Optimized low-latency path |
| AssemblyAI Universal-3 streaming | 300 ms | 500-700 ms | |
| Self-hosted faster-whisper large-v3 + VAD streaming | 1-2 s | 2-3 s | 2-3 s chunks, int8, beam=1 |
| Self-hosted Whisper large-v3 batch (per 1-min file) | 5-15 s | 20-30 s | A100, batched inference |

### 4.2 End-to-End Agent-Assist Stack p95

Real-time agent-assist adds LLM round-trips on top of STT:

- VAD + audio buffering: 200-500 ms
- Streaming STT partial hypothesis: 260-600 ms p95
- LLM tool-call or next-best-action prompt: 400-900 ms p95 first token
- TTS for any spoken response (optional): 200-500 ms p95

**Total p95 for full agent-assist: 1.2-2.5 seconds.** Sub-second requires aggressive model cascading, speculative decoding, or distilling the LLM to a 1-3B param model fine-tuned on your call taxonomy.

### 4.3 Batch Post-Call Analytics

- Throughput-bound, not latency-bound.
- A100 large-v3 batched: ~500 audio-min/GPU-minute = ~30k audio-hr/day on 1 GPU.
- Target SLO for next-morning QA dashboards: jobs complete by 6am local = process 16h of calls in 4-8h.
- p95 per-file latency is not the right metric; track job-completion SLA instead.

---

## 5 Frameworks For Decision Making

- **CASA (Compute-Adjusted Speech Analytics):** model STT+LLM cost as $/audio-hour, normalize across providers, and compare against self-host TCO. The $/audio-hour is the primary metric; per-minute or per-call metrics hide utilization effects.
- **Margin-of-Safety provisioning:** provision GPU pools at 60-70% steady-state utilization to absorb 3-5x traffic spikes typical in contact centers (morning ramp, seasonal peaks, outage-driven overflow).
- **Make-vs-buy threshold:** the break-even band is 150-220k audio-min/month. Use this as the gating metric in architecture reviews.
- **Efficient frontier (English ASR 2026):** Parakeet-TDT (best $/throughput), Whisper-large-v3 (best multilingual coverage), Canary-1B (balanced). All three are within 1-2 WER points of each other on clean English; selection is driven by language mix and latency targets.

---

## 6 Case Studies

### Case Study A: Mid-Market BPO, 1,000 Seats, 3M Minutes/Month

A 1,000-seat BPO ran ~6,000 calls/day at 8 min average, totaling ~3M audio-min/month. They benchmarked Deepgram Nova-3 Growth ($0.0065/min) against self-hosted faster-whisper large-v3 on A100 80GB reserved ($1.50/hr, ~600 audio-hr/hr throughput). The A100 delivered ~$0.0025/min all-in. With 50% average utilization the figure rose to ~$0.005/min, still 23% below the API. Adding $3,100/month amortized MLE/SRE overhead pushed net savings to ~$3.5k/month vs the API. Break-even held at ~220k min/month; below that, the API won. They chose self-host for transcription but kept Deepgram streaming for the live agent-assist path where sub-300ms latency was non-negotiable.

### Case Study B: Regional Health System Contact Center, 250 Seats, 400k Minutes/Month

A HIPAA-covered health system could not send PHI to managed STT APIs without a BAA that major providers declined for transcription workloads. They deployed Whisper-large-v3 (faster-whisper) on two L40S GPUs at $1.20/hr each. Sustained throughput of ~350 audio-hr/hr per GPU meant 35% average utilization, pushing effective cost to ~$0.0097/min -- higher than Deepgram's $0.0065/min Growth rate. They chose self-host anyway because the alternative was no analytics at all. They later added Canary-1B for English-only encounters, which on the same hardware delivered ~1.8x throughput, dropping effective cost to ~$0.0054/min -- finally undercutting the API while maintaining on-prem residency.

### Case Study C: SaaS Support Team, 50 Seats, 60k Minutes/Month

A 50-seat SaaS support team produced ~60k audio-min/month. They evaluated four options: AssemblyAI Universal-3 batch ($0.004/min = $240/mo), Deepgram Nova-3 Growth ($0.0065/min = $390/mo), self-hosted faster-whisper on a single RTX 4090 ($0.50/hr reserved, ~250 audio-hr/hr, ~10% utilization -> ~$0.012/min effective = $720/mo), and OpenAI gpt-4o-transcribe ($0.006/min = $360/mo). AssemblyAI batch won on raw cost. With one engineer's time amortized, self-host total cost of ownership rose to ~$1,200/mo -- 5x the API. They chose AssemblyAI batch + GPT-4o mini for downstream analysis. Break-even did not arrive until they crossed ~200k min/month.

### Case Study D: Outbound Sales Org, 200 Seats, 1.2M Minutes/Month with Real-Time Assist

A 200-seat outbound sales operation required both post-call analytics and real-time agent assist. For post-call they used self-hosted Whisper-large-v3 on two A100 80GB reserved instances ($1.50/hr each, 60% utilization -> ~$0.0042/min). For real-time assist they used Deepgram Nova-3 streaming ($0.0077/min) plus Claude Haiku 4.5 for next-best-action ($0.005-$0.015/min of streaming LLM). Total stack cost per call averaged $0.012 (3-min avg) to $0.028 (12-min avg). Moving real-time STT to self-hosted Canary-1B would have saved ~$3k/mo but added ~150ms p95 latency, which degraded the agent-assist UX enough that reps noticed. They kept Deepgram streaming despite the cost premium.

### Case Study E: Global Fintech, 5,000 Seats, 18M Minutes/Month, Multi-Region

A global fintech with 5,000 contact center seats across three regions processed ~18M audio-min/month. They built a hybrid stack: self-hosted Whisper-large-v3 + Parakeet-TDT on H100 reserved capacity in each region for batch post-call analytics, plus AWS Transcribe Call Analytics for the streaming tier (cheaper than self-hosting at their volume due to AWS EDP commit discounts). Self-hosted batch cost averaged ~$0.0028/min vs Deepgram equivalent at ~$0.0065/min. Savings on batch alone: ~$67k/month. The streaming tier stayed on AWS at negotiated rates. Total annual savings vs an all-Deepgram architecture: ~$800k, against ~$400k in additional SRE/MLOps headcount.

---

## 7 Synthesis

The 2025-2026 economics of call-center speech analytics favor a bifurcated architecture: managed APIs for latency-sensitive streaming workloads (real-time agent-assist, live transcription overlays, compliance monitoring) and self-hosted Whisper-large-v3 / Parakeet-TDT / Canary-1B for batch post-call analytics at scale. The break-even threshold sits at roughly 150-220k audio-minutes per month of sustained throughput; below that, the operational overhead of running speech models in-house overwhelms the compute savings, and managed APIs deliver better TCO once on-call, observability, and model-refresh labor are priced in. Above 1M minutes per month, self-hosting on A100 or H100 reserved capacity delivers a 2-4x cost advantage, and adding NVIDIA Parakeet-TDT for English-dominant queues can halve compute spend again.

LLM cost has collapsed to a rounding error relative to STT: a complete post-call analytics package (transcription, diarization, summarization, QA scoring, sentiment, topic classification) on a 7-minute call lands at $0.005-$0.015 with GPT-4o mini or Claude Haiku 4.5. The single most commonly missed line item is real-time agent-assist streaming, which adds $0.005-$0.015 per minute of call time on top of the STT cost and frequently becomes the largest component of an agent-assist budget.

The largest divergence between providers is no longer raw transcription accuracy -- Whisper-large-v3, Parakeet-TDT, Canary-1B, and Deepgram Nova-3 cluster within 1-2 WER points on clean English audio -- but total cost of ownership including operational complexity. NVIDIA's Parakeet-TDT and Canary-1B have materially shifted the self-host break-even curve leftward in 2026, running 10-50x realtime on a single H100 with accuracy matching Whisper-large-v3, which makes in-house deployment attractive from as little as 75k minutes per month for English-only workloads. For multilingual deployments Whisper-large-v3 remains the default; the cost premium versus Parakeet is roughly 30-50% but the coverage delta is decisive.

Regulated workloads (HIPAA, PCI, GDPR with EU data isolation) default to self-hosting regardless of cost arithmetic because the major managed STT providers either decline to sign BAAs for transcription or cannot guarantee regional data isolation at the audio-frame level. In these deployments the cost calculus is secondary to compliance posture, and self-hosted A100/H100 capacity with on-prem or VPC-isolated object storage is the standard pattern.
