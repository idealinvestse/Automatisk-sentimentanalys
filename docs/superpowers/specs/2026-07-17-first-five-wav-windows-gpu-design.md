# First five WAV test — Windows + RTX 5070 design

**Date:** 2026-07-17  
**Status:** Draft for user review  
**Approach:** 1 — Native Windows + launcher (approved)  
**Host:** Windows 10/11, NVIDIA RTX 5070 12 GB VRAM  
**Audio:** 5 Swedish demo `.wav` files (safe demos; must remain gitignored)

---

## Problem frame

The repo is ready for a controlled first host test, but sample audio is missing, `sv_callcenter` is disabled in the manifest, and there is no documented A→B→C path tailored to a 12 GB Blackwell GPU with automatic large→medium ASR fallback on CUDA OOM. Agents also lack project-scoped subagents for this pilot loop.

## Goals

1. Prepare the Windows host so five local `.wav` files can be processed end-to-end.
2. Run tests in order: **A** ASR-smoke → **B** full pipeline (OpenRouter) → **C** API + webui.
3. Prefer `kb-whisper-large`; on CUDA OOM, retry once with `kb-whisper-medium` and log the fallback.
4. Keep demos under `samples/audio/sv/callcenter/` while ensuring audio binaries stay out of git.
5. Add Cursor project subagents that encode the pilot checklist and policy guards.

## Non-goals

- Docker GPU path for this first test
- Cloud STT / Deepgram / Groq as defaults
- DATA-01 corpus gates or production quality claims
- Full `API_PRODUCTION=true` hardening (optional later; not required for first five-file smoke)

---

## Decisions (locked)

| Decision | Choice |
|----------|--------|
| Audio source | User-supplied 5 Swedish demo `.wav` |
| Depth | A → B → C in order |
| ASR model | Auto: large first, medium on OOM |
| LLM | `OPENROUTER_API_KEY` present; OpenRouter → Mistral |
| Runtime | Native Windows + launcher |
| PII | Safe demos; path gitignored (already `samples/audio/**/*.wav`) |

---

## Architecture / flow

```text
[5× .wav] → samples/audio/sv/callcenter/  (gitignore)
                ↓
        manifest: sv_callcenter enabled=true
                ↓
A) python -m src.evaluate audio smoke --pack sv_callcenter --device cuda
   ASR: kb-whisper-large → CUDA OOM → kb-whisper-medium (one retry)
                ↓
B) python -m src.evaluate audio run --scenario pipeline --pack sv_callcenter
   profile=callcenter; LLM via OpenRouter; anonymize_before_llm=true
                ↓
C) launcher: API :8000 + webui :3000
   Manual: Transkribering / Testlabb with same files
```

### Components

| Piece | Role |
|-------|------|
| `scripts/dev-setup.ps1`, `launcher.ps1` | venv, deps, doctor, start/stop |
| `configs/install_defaults.yaml` + user_config | device, ASR model, LLM provider |
| `samples/audio/manifest.yaml` | enable `sv_callcenter` |
| `src.evaluate audio *` | list / validate / smoke / pipeline |
| OOM fallback helper | thin wrapper or eval flag (not present today as auto) |
| `.cursor/agents/*` | windows-gpu-pilot, asr-smoke-runner, pilot-policy-guard |

### Environment

- `OPENROUTER_API_KEY` in `.env` (gitignored)
- ASR `provider=local` only
- Do not set / use `DEEPGRAM_API_KEY` or `GROQ_API_KEY` for this test
- Optional: `HF_TOKEN` if pyannote diarization is desired; otherwise heuristic fallback is acceptable

---

## Audio pack layout

```
samples/audio/sv/callcenter/
  01_*.wav … 05_*.wav
  (optional) matching *.meta.yaml sidecars
```

- Enable pack: `sv_callcenter.enabled: true` in `samples/audio/manifest.yaml`
- Validate: `python -m src.evaluate audio validate`
- Confirm gitignore still covers `samples/audio/**/*.wav` (already present)

Optional sidecars (recommended for B evaluation clarity):

```yaml
expected_sentiment: negativ
scenario: billing_complaint
speakers: 2
notes: "Demo — no real customer PII"
```

---

## OOM fallback behavior

1. Attempt load/transcribe with `kb-whisper-large`, `device=cuda`, float16/default compute type.
2. On CUDA out-of-memory (or equivalent torch/cublas OOM): free GPU memory, switch model to `kb-whisper-medium`, retry **once**.
3. Log clearly: original model, fallback model, which file failed.
4. If medium also OOMs: fail the run with an actionable message (e.g. disable diarization / reduce batch / use preprocess).
5. In “GPU test” mode: do **not** silently fall back to CPU.

Implementation detail is deferred to the plan (script vs evaluate flag vs small library helper). Direction: keep change minimal and reusable from CLI and subagents.

---

## Verification / success criteria

1. Doctor/status OK: Python, ffmpeg, CUDA device visible as RTX 5070 (or equivalent name).
2. `evaluate audio list --pack sv_callcenter` shows 5 files.
3. **A:** Smoke completes on CUDA (large or medium after fallback); transcripts produced.
4. **B:** Pipeline yields sentiment; LLM path works with OpenRouter; no cloud-STT egress.
5. **C:** `GET /health` OK; webui loads; at least one file run via Transkribering or Testlabb.

### Error handling matrix

| Problem | Action |
|---------|--------|
| CUDA OOM | Unload → medium → one retry |
| No CUDA | Fail clearly (no silent CPU in GPU-test mode) |
| Missing OpenRouter key | Allow A; block B deep-path with instructions |
| Diarization OOM/error | Heuristic fallback; note in test report |

---

## Subagents (project scope)

Create under `.cursor/agents/`:

1. **windows-gpu-pilot** — Host provision, CUDA check, A→B→C orchestration checklist for this design.
2. **asr-smoke-runner** — Pack validation, OOM-aware smoke commands, interpret evaluate output.
3. **pilot-policy-guard** — Enforce local ASR, block Groq/Deepgram defaults, callcenter anonymize flags.

Descriptions must include proactive trigger language so agents delegate when the user starts a five-file / Windows GPU pilot.

---

## Deliverables (implementation plan → work)

1. This design spec (review gate).
2. Implementation plan via writing-plans / ce-plan.
3. Short operator runbook: `docs/FIRST_FIVE_WAV_TEST.md`.
4. OOM fallback helper (minimal).
5. Manifest enablement + any config defaults needed for CUDA/OpenRouter on Windows.
6. Project subagent markdown files.
7. Smoke verification commands documented; optional script that runs A then prompts for B/C.

---

## Risks

| Risk | Mitigation |
|------|------------|
| Blackwell (5070) needs newer CUDA/torch than docs assume | Doctor step prints torch/CUDA versions; pin/install guidance in runbook |
| `kb-whisper-large` + diarization exceeds 12 GB | OOM fallback; diarization optional for A |
| Demo files mis-committed | Rely on existing gitignore; verify `git status` after copy |
| OpenRouter rate/cost | Soft budget already in config; limit=5 |

---

## Approval record

- Approach 1 (native Windows + launcher): approved
- Design §1 scope: approved
- Design §2 flow: approved
- Design §3 verification + subagents: approved
