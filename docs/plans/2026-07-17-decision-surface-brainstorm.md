---
name: decision-surface-executive-pack
date: 2026-07-17
artifact_readiness: requirements-only
status: locked
---

# Decision surface — Automatisk-sentimentanalys

Locked from approved plan + repo audit (no further product interview required for this report pack).

## Product contract (decision pack scope)

**Goal:** One executive decision pack that answers pilot go/no-go, deployment/compliance posture, and 90-day tech priority — not a feature wishlist.

**Primary buyer situation:** Nordic contact-center / QA leads who need Swedish-capable speech analytics with EU-aligned data handling, without adopting a full international suite.

**Success criteria for this pack:**
1. Clear pilot verdict: go / conditional go / no-go
2. Locked recommendations for ASR cloud, Groq, OpenRouter/Mistral, local-first default
3. Ranked P0/P1/P2 for next 90 days tied to real gaps (DATA-01, eval, L7–L9, ops)
4. External claims backed by Parallel research artifacts

## Decisions that must be resolved in the report

| Decision | Options considered |
|----------|-------------------|
| Pilot readiness | go / conditional go / no-go |
| Deployment model | local-only / hybrid (local ASR + selective EU LLM) / cloud-heavy |
| Deepgram cloud STT | never for PII / opt-in with DPA / prod default |
| Groq | forbidden in prod / dev-only / prod with anonymize gate |
| DATA-01 corpus | min size + ownership before quality gates move |
| Intent backend | stay heuristic until +0.05 F1 vs model |
| Positioning | Swedish+EU residency differentiator vs feature-parity race |

## Non-goals for this pack

- Implementing code or DATA-01 import
- Running L7–L9 on GPU hardware in this session
- Reopening YouTube ingest (Fas 5)
- Full WS-first product rewrite

## Research questions (Parallel)

1. Nordic/EU CC intelligence market & competitors
2. GDPR/DPA for hybrid ASR+LLM (Deepgram, OpenRouter, Groq, HF)
3. Cost/latency SLOs and GPU break-even
4. Swedish ASR/sentiment/intent production baselines
