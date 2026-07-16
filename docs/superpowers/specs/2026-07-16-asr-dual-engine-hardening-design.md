# ASR dual-engine hardening design

**Date:** 2026-07-16  
**Status:** Draft for user review  
**Approach:** C — Dual-engine early (approved)  
**Goals:** Helhetspaket — kvalitet + robusthet + drift/skala  
**Surfaces:** Shared core in `src/transcription/`; Windows launcher + Docker/API both consume it  
**Data policy:** Local ASR default; cloud STT opt-in only (explicit flag + documented risk)

**Research input:** Parallel.ai deep research `asr-pipeline-robustness-2026` (repo root of inner project).

---

## Problem frame

The product already has a solid local Whisper stack (`faster-whisper` + KB-Whisper, preprocess/VAD, chunking, job progress over WebSocket). Gaps that hurt production sentiment quality and ops:

- Whisper silence/repetition hallucinations reach analyzers
- Chunk failures are skipped without retry; partial holes are silent
- `add_diarization()` hardcodes heuristic even when pyannote is available
- Transcription jobs are in-memory only (lost on API restart)
- No first-class way to A/B local vs commercial STT under the same contract
- Weak ASR observability (latency/fail/cloud egress) for SLOs

Downstream sentiment/intent is only as good as the transcript. Hardening ASR and adding a controlled cloud path improves the whole pipeline without rewriting analyzers.

---

## Goals

1. **Quality** — Fewer ghost segments; clearer confidence gates; better call-center defaults; honest diarization when token present.
2. **Robustness** — Chunk retry, stable error codes, no silent cloud fallback, preprocess/diarize graceful degrade unchanged.
3. **Ops** — Persistent jobs, structured metrics, launcher/API parity via shared router.
4. **Dual-engine** — Cloud STT behind the same `Transcriber` protocol from the start; local remains default.

## Non-goals

- Realtime streaming partial ASR (WebSocket stays job progress/logs)
- Custom Whisper fine-tuning
- Multiple cloud providers in v1 (Deepgram first; interface ready for more)
- Edge offline ASR (edge remains text-only)
- Changing sentiment analyzer algorithms (only consume better transcripts / confidence)

---

## Architecture

```text
Audio → preprocess (off|basic|callcenter)
      → AsrRouter (provider=local|cloud; never silent cloud)
           ├─ LocalWhisperEngine (faster | transformers | whisperx)
           └─ CloudSttEngine (Deepgram v1; more later)
      → shared post-process (hallucination filter, confidence flags)
      → optional diarization (pyannote if HF_TOKEN + diarize; else heuristic)
      → Transcript → pipeline / API / webui
```

| Component | Responsibility |
|-----------|----------------|
| `Transcriber` protocol | Existing contract: `transcribe(...) → Transcript` |
| `AsrRouter` (new) | Selects engine from config; enforces local default and cloud policy |
| `LocalWhisperEngine` | Existing backends + local harden (decode flags, chunk retry) |
| `CloudSttEngine` (new) | Opt-in HTTP STT → same `Transcript` shape |
| Shared post-process | Hallucination filter, confidence normalization, metadata/`warnings[]` |
| Job store | Persist status beyond process memory |
| Evaluate compare | A/B local vs cloud on audio manifest |

**Hard policy**

- Default: `asr.provider=local` (or equivalent config key).
- Cloud requires explicit `provider=cloud` + API key + documented risk in SECURITY/docs.
- Raw audio egress logged as security event (`asr_cloud_egress=true`) without logging audio or transcript body.
- No silent fallback from cloud → local unless `asr.cloud_fallback_local=true` (default **false**).

Factory/cache today (`get_transcriber` + `@lru_cache`) remains for local backends; router sits above or wraps factory so callers (pipeline, API helpers, CLI) do not choose engines ad hoc.

---

## Local hardening

1. **Decode safety** — Prefer `condition_on_previous_text=False` for call-center/long audio (configurable). Tighten no-speech / logprob handling where faster-whisper exposes it.
2. **Hallucination post-filter** — Drop or flag known ghost patterns (e.g. “Thanks for watching”, empty/near-empty with high repetition, speech on near-silence). Apply after ASR, before analyzers.
3. **Chunk robustness** — On per-chunk failure: retry 1–2 times; if still failing, record warning and continue (no silent hole without metadata). Keep overlap merge; prefer higher-confidence overlap text.
4. **Confidence contract** — Keep `low_confidence` (avg word prob &lt; 0.60); ensure metadata is consistent across backends; allow downstream LLM-judge / lexicon boost to key off it (existing path).
5. **Call-center profile** — When profile is call-center, recommend `preprocess_mode=callcenter` (docs + UI); do not force globally.
6. **Diarization** — When `diarize=true` and `HF_TOKEN`/`HUGGINGFACE_HUB_TOKEN` present, use pyannote path; heuristic remains fallback. Fix `add_diarization()` so it does not hardcode heuristic-only.

---

## Cloud adapter (Deepgram first)

- Implements `Transcriber` protocol; maps provider response → `Segment` / `Word` / confidence.
- Timeouts; exponential backoff only on 429 / 5xx / network errors.
- Auth failures and quota → fail with stable error codes (no retry storm).
- Secrets via env (e.g. `CLOUD_STT_API_KEY` / provider-specific); never committed.
- Launcher/webui: explicit toggle + warning copy; install does not download cloud SDKs as required core deps (optional extra, e.g. `[cloud-stt]`).

**Error codes (stable strings)**

| Code | Meaning |
|------|---------|
| `MODEL_LOAD` | Local model failed to load |
| `CHUNK_FAILED` | Chunk exhausted retries |
| `CLOUD_AUTH` | Cloud credentials rejected |
| `CLOUD_TIMEOUT` | Cloud request timed out |
| `CLOUD_QUOTA` | Rate limit / quota |
| `PREPROCESS_FAILED` | Warn + continue with original audio (existing) |
| `DIARIZE_FAILED` | Warn + continue without speakers (existing) |

Pipeline may still return empty transcript + error metadata on fatal ASR (current behavior); API docs clarify 5xx vs partial success with `warnings`.

---

## Jobs, metrics, surfaces

**Jobs**

- Keep sync `/transcribe` for simple cases.
- Background jobs: persist registry (SQLite **or** Redis; choose at implementation by what API already uses for WS fan-out — prefer Redis when `API_USE_REDIS_CACHE`, else SQLite file under data root).
- Preserve cancel, progress, WS event types; hub writes through store.
- Batch: per-file timeout + retry policy (cloud transient vs local hard fail).

**Metrics / logging**

Structured fields: `provider`, `backend`, `model`, `duration_s`, `processing_s`, `low_conf_ratio`, `hallucination_drops`, `error_code`.  
Expose via existing `/metrics` where practical: ASR latency p50/p95, fail-rate, cloud call count.

**Surfaces**

- CLI / API / launcher / webui: only config + warnings; no duplicate ASR logic.
- `configs/install_defaults.yaml` + `AsrDefaults`: add `provider`, cloud flags; default local.

---

## Evaluation & A/B

Extend `src/benchmarks/` / `sentimentanalys evaluate audio` with provider dimension:

- Same `samples/audio` manifest (or documented subset)
- Report WER (when reference exists), latency, estimated cloud cost
- Example: `evaluate audio compare --providers local,cloud`
- Live cloud tests marked `@pytest.mark.cloud_stt`; never required in default CI

---

## Testing strategy

| Layer | Coverage |
|-------|----------|
| Unit | Hallucination filter; router defaults; cloud response mapper (fixtures); no network |
| Unit/integration | Chunk retry warnings; diarization backend selection with/without token (mocked) |
| Existing | Extend `tests/test_asr.py`, job tests for persistence |
| Audio | `@pytest.mark.audio` local smoke; L7 checklist remains |
| Cloud live | Opt-in marker + env key only |

---

## Documentation & security

- `SECURITY.md`: cloud STT opt-in — what leaves the machine, how to disable, logging rules.
- `docs/API.md` / `ARCHITECTURE.md`: provider, error codes, job persistence.
- `docs/WINDOWS_INSTALL.md` / launcher copy: toggle warning.
- Research artifact paths (optional reference): `asr-pipeline-robustness-2026.md` / `.json` at inner repo root (not product docs).

---

## Phased delivery

| Phase | Deliverable |
|-------|-------------|
| **P0** | `AsrRouter` + config/policy (local default); hallucination filter; decode harden; chunk retry + warnings |
| **P1** | Deepgram `CloudSttEngine`; A/B compare CLI; SECURITY/docs; optional `[cloud-stt]` |
| **P2** | Persistent transcription jobs; ASR metrics fields on `/metrics` |
| **P3** | Diarization uses pyannote when token present; call-center preprocess recommendation in UI/docs |

P0 and P1 may proceed in parallel after router/config lands (cloud adapter does not block local filter work once protocol + router exist).

---

## Success criteria

- Local: measurable drop in ghost segments on silent/noisy fixtures; chunk failures produce retry and/or explicit `warnings[]`, not silent gaps.
- Cloud: end-to-end opt-in path works; default CI green without cloud credentials; no cloud call without explicit config.
- Ops: job status survives API process restart (chosen store); metrics show provider/latency/fail-rate.
- Parity: launcher and API both go through router; no divergent ASR business logic in UI layers.

---

## Key decisions (locked)

| Decision | Choice |
|----------|--------|
| Overall approach | C — dual-engine early |
| Default engine | Local (`faster` + `kb-whisper-large`) |
| Cloud | Opt-in; Deepgram first |
| Cloud → local fallback | Off by default |
| Shared core | `src/transcription/` |
| Streaming ASR | Out of scope |
| Commit of secrets | Forbidden; env only |

---

## Open points deferred to implementation plan

- Exact SQLite vs Redis job-store wiring details
- Precise Deepgram model/API version and feature flags (nova vs enhanced)
- Exact regex/heuristic list for hallucination filter (seed from research + Swedish call fixtures)
- Whether `condition_on_previous_text` default differs by profile only or globally

These do not change the product contract above.
