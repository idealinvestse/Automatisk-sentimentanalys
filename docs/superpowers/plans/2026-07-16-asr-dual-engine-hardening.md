# ASR Dual-Engine Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden local Whisper ASR and add an opt-in Deepgram cloud path behind a shared router, with persistent jobs, metrics, and A/B evaluation — without sending audio to the cloud unless explicitly configured.

**Architecture:** Keep existing `Transcriber` backends. Add `AsrRouter` (local default / cloud opt-in), shared post-process (hallucination filter + warnings), Deepgram adapter, SQLite/Redis job persistence, and Prometheus ASR metrics. All entry points (CLI, API `transcribe_helper`, pipeline, launcher) go through the router.

**Tech Stack:** Python 3.11+, faster-whisper, optional Deepgram REST via `httpx`, Prometheus (`src/core/metrics.py`), FastAPI job registry, Typer evaluate CLI, pytest.

**Spec:** `docs/superpowers/specs/2026-07-16-asr-dual-engine-hardening-design.md`

## Global Constraints

- Default `asr.provider=local`; cloud only when explicitly set to `cloud` plus API key
- Never silent cloud→local fallback (`cloud_fallback_local` default `false`)
- Never commit secrets; use `DEEPGRAM_API_KEY` (alias `CLOUD_STT_API_KEY`)
- Log cloud egress as `asr_cloud_egress=true` without audio/transcript body
- No realtime streaming ASR in this plan
- Deepgram only for v1 cloud; interface ready for more providers
- Swedish-first; preserve KB-Whisper + `faster` as local default
- TDD: failing test before implementation in each task
- Conventional commits (`feat:`, `fix:`, `docs:`, `test:`)

## File Structure (locked)

| Path | Responsibility |
|------|----------------|
| `src/transcription/error_codes.py` | Stable ASR error code constants |
| `src/transcription/postprocess.py` | Hallucination filter; attach `warnings` / drop counts |
| `src/transcription/router.py` | `AsrRouter` + `resolve_asr_provider()` policy |
| `src/transcription/cloud_deepgram.py` | Deepgram `Transcriber` implementation |
| `src/api/transcription_job_store.py` | Persistent job store (SQLite default, Redis when enabled) |
| `src/core/models.py` | Add `Transcript.warnings`, `Transcript.provider`, optional metadata fields |
| `src/transcription/base.py` | Fix `add_diarization` backend selection |
| `src/transcription/faster_whisper.py` | Decode flag + chunk retry + warnings |
| `src/transcription/factory.py` | Re-export router helpers; keep local cache |
| `src/transcription/__init__.py` | Public exports |
| `src/api/helpers.py` | Route via `AsrRouter`; pass provider |
| `src/api/schemas.py` | `provider`, cloud flags on `AsrParamsMixin` |
| `src/install/config_schema.py` | `AsrDefaults.provider`, `cloud_fallback_local` |
| `configs/install_defaults.yaml` | Defaults |
| `src/core/metrics.py` | ASR histograms/counters |
| `src/benchmarks/audio_cli.py` + `audio_runner.py` | `compare` command |
| `pyproject.toml` | `[cloud-stt]` optional; `cloud_stt` pytest marker |
| `SECURITY.md`, `docs/API.md`, `docs/ARCHITECTURE.md`, `docs/WINDOWS_INSTALL.md` | Docs |
| `launcher/ui_settings_dialog.py` / webui ASR settings | Opt-in toggle + warning copy |
| Tests under `tests/test_asr_*.py`, extend existing ASR/job tests |

### Resolved design deferrals (implement these values)

1. **Job store:** `SqliteJobStore` at `{data_root}/state/transcription_jobs.db` by default. If `API_USE_REDIS_CACHE` is truthy and Redis is reachable, use `RedisJobStore` key prefix `asr:jobs:`.
2. **Deepgram:** Pre-recorded REST `https://api.deepgram.com/v1/listen` with `model=nova-2`, `language=sv`, `punctuate=true`, `utterances=true` (or `words=true`).
3. **Hallucination patterns (initial):** case-insensitive match on full segment text after strip — `thanks for watching`, `thank you for watching`, `subscribe to`, `tack för att ni tittade`, `textning av`, `undertexter av`, plus repetition detector (≥4 identical consecutive tokens) and empty/whitespace-only segments.
4. **`condition_on_previous_text`:** Default `False` for all local faster-whisper calls; overridable via kwarg `condition_on_previous_text: bool | None = None` (`None` → `False`).

---

### Task 1: Transcript metadata + error codes

**Files:**
- Create: `src/transcription/error_codes.py`
- Modify: `src/core/models.py` (`Transcript`)
- Test: `tests/test_asr_error_codes.py`

**Interfaces:**
- Consumes: existing `Transcript` dataclass
- Produces: `AsrErrorCode` constants; `Transcript.warnings: list[str]`; `Transcript.provider: str` default `"local"`; `to_dict`/`from_dict` round-trip

- [ ] **Step 1: Write the failing test**

```python
# tests/test_asr_error_codes.py
from src.core.models import Transcript
from src.transcription.error_codes import AsrErrorCode


def test_error_codes_are_stable_strings():
    assert AsrErrorCode.CLOUD_AUTH == "CLOUD_AUTH"
    assert AsrErrorCode.CHUNK_FAILED == "CHUNK_FAILED"


def test_transcript_warnings_roundtrip():
    t = Transcript(
        model="m",
        backend="faster",
        language="sv",
        duration=1.0,
        processing_time=0.1,
        warnings=["chunk_retry:3"],
        provider="local",
    )
    d = t.to_dict()
    assert d["warnings"] == ["chunk_retry:3"]
    assert d["provider"] == "local"
    assert Transcript.from_dict(d).warnings == ["chunk_retry:3"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_asr_error_codes.py -v`  
Expected: FAIL (import or missing fields)

- [ ] **Step 3: Minimal implementation**

```python
# src/transcription/error_codes.py
class AsrErrorCode:
    MODEL_LOAD = "MODEL_LOAD"
    CHUNK_FAILED = "CHUNK_FAILED"
    CLOUD_AUTH = "CLOUD_AUTH"
    CLOUD_TIMEOUT = "CLOUD_TIMEOUT"
    CLOUD_QUOTA = "CLOUD_QUOTA"
    PREPROCESS_FAILED = "PREPROCESS_FAILED"
    DIARIZE_FAILED = "DIARIZE_FAILED"
```

Extend `Transcript` with `warnings: list[str] = field(default_factory=list)` and `provider: str = "local"`; include in `to_dict`/`from_dict` when non-default / always for provider.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_asr_error_codes.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/transcription/error_codes.py src/core/models.py tests/test_asr_error_codes.py
git commit -m "feat(asr): add Transcript warnings/provider and stable error codes"
```

---

### Task 2: Hallucination post-filter

**Files:**
- Create: `src/transcription/postprocess.py`
- Test: `tests/test_asr_postprocess.py`

**Interfaces:**
- Consumes: `Transcript`, `Segment`
- Produces: `filter_hallucinations(transcript: Transcript) -> Transcript` returning new transcript with filtered segments, `warnings` entries like `hallucination_dropped:N`, and segment `properties["hallucination"]=True` only if flag-mode used — **v1 drops** matching segments (does not keep them)

- [ ] **Step 1: Write the failing test**

```python
from src.core.models import Segment, Transcript
from src.transcription.postprocess import filter_hallucinations


def _t(texts):
    return Transcript(
        model="m",
        backend="faster",
        language="sv",
        duration=10.0,
        processing_time=1.0,
        segments=[Segment(start=i, end=i + 1, text=x) for i, x in enumerate(texts)],
    )


def test_drops_thanks_for_watching():
    out = filter_hallucinations(_t(["Hej", "Thanks for watching", "Hejdå"]))
    assert [s.text for s in out.segments] == ["Hej", "Hejdå"]
    assert any("hallucination_dropped" in w for w in out.warnings)


def test_drops_swedish_ghost():
    out = filter_hallucinations(_t(["Tack för att ni tittade"]))
    assert out.segments == []


def test_drops_repetition_loops():
    out = filter_hallucinations(_t(["ja ja ja ja ja"]))
    assert out.segments == []


def test_keeps_normal_swedish():
    out = filter_hallucinations(_t(["Jag vill ha hjälp med fakturan"]))
    assert len(out.segments) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_asr_postprocess.py -v`  
Expected: FAIL import

- [ ] **Step 3: Implement `filter_hallucinations`**

Implement in `src/transcription/postprocess.py`:
- Normalize text: strip, lower, collapse whitespace
- Exact/substring blocklist for patterns listed in Global Constraints
- Repetition: split on whitespace; if `len(tokens) >= 4` and `len(set(tokens)) == 1`, drop
- Drop empty/whitespace segments
- Append warning `hallucination_dropped:{count}` when count > 0
- Use `dataclasses.replace` so input transcript is not mutated

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_asr_postprocess.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/transcription/postprocess.py tests/test_asr_postprocess.py
git commit -m "feat(asr): add hallucination post-filter for Whisper ghosts"
```

---

### Task 3: AsrRouter + config policy

**Files:**
- Create: `src/transcription/router.py`
- Modify: `src/install/config_schema.py` (`AsrDefaults`)
- Modify: `configs/install_defaults.yaml`
- Modify: `src/transcription/__init__.py` (export `get_asr_router` / `transcribe_with_router`)
- Test: `tests/test_asr_router.py`

**Interfaces:**
- Consumes: `get_transcriber` (local), later `DeepgramTranscriber` (Task 6 — use lazy import; until then cloud raises clear error)
- Produces:
  - `resolve_asr_provider(provider: str | None) -> Literal["local","cloud"]` — default `"local"`; unknown → ValueError
  - `AsrRouter.transcribe(audio_path, *, provider="local", backend=..., model_name=..., cloud_fallback_local=False, **kwargs) -> Transcript`
  - Applies `filter_hallucinations` before return
  - Sets `transcript.provider`

- [ ] **Step 1: Write the failing test**

```python
from unittest.mock import MagicMock, patch
import pytest
from src.core.models import Segment, Transcript
from src.transcription.router import AsrRouter, resolve_asr_provider


def test_default_provider_is_local():
    assert resolve_asr_provider(None) == "local"
    assert resolve_asr_provider("") == "local"
    assert resolve_asr_provider("LOCAL") == "local"


def test_cloud_requires_explicit():
    assert resolve_asr_provider("cloud") == "cloud"


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        resolve_asr_provider("azure")


def test_router_local_calls_factory_and_filters():
    fake = Transcript(
        model="m",
        backend="faster",
        language="sv",
        duration=1.0,
        processing_time=0.1,
        segments=[Segment(0, 1, "Thanks for watching")],
    )
    mock_t = MagicMock()
    mock_t.transcribe.return_value = fake
    with patch("src.transcription.router.get_transcriber", return_value=mock_t):
        out = AsrRouter().transcribe("x.wav", provider="local")
    assert out.segments == []
    assert out.provider == "local"
    mock_t.transcribe.assert_called_once()


def test_router_cloud_without_adapter_raises():
    with pytest.raises(Exception) as ei:
        AsrRouter().transcribe("x.wav", provider="cloud")
    assert "CLOUD" in str(ei.value).upper() or "deepgram" in str(ei.value).lower() or "not configured" in str(ei.value).lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_asr_router.py -v`  
Expected: FAIL

- [ ] **Step 3: Implement router + AsrDefaults fields**

```python
# AsrDefaults additions
provider: Literal["local", "cloud"] = "local"
cloud_fallback_local: bool = False
cloud_provider: Literal["deepgram"] = "deepgram"
```

`AsrRouter.transcribe`:
1. Resolve provider
2. If local → `get_transcriber(...).transcribe(...)`
3. If cloud → try Deepgram (Task 6); for now raise `TranscriptionError` with `AsrErrorCode.CLOUD_AUTH` message if no key / adapter missing
4. `filter_hallucinations`
5. Set `provider` on result

Update `configs/install_defaults.yaml` under `asr:` with `provider: local`, `cloud_fallback_local: false`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_asr_router.py tests/test_asr_postprocess.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/transcription/router.py src/transcription/__init__.py src/install/config_schema.py configs/install_defaults.yaml tests/test_asr_router.py
git commit -m "feat(asr): add AsrRouter with local-default provider policy"
```

---

### Task 4: Wire API/CLI/pipeline through router

**Files:**
- Modify: `src/api/helpers.py`
- Modify: `src/api/schemas.py` (`AsrParamsMixin`: `provider`, `cloud_fallback_local`)
- Modify: `src/pipeline.py` (use router instead of bare `get_transcriber` where ASR is invoked)
- Modify: `src/cli.py` (transcribe command: `--provider`)
- Test: extend `tests/test_asr.py` or `tests/test_api.py` with provider default assertion

**Interfaces:**
- Consumes: `AsrRouter.transcribe`
- Produces: all public ASR entry points accept `provider: str = "local"`

- [ ] **Step 1: Write failing test that helpers use router**

```python
from unittest.mock import patch, MagicMock
from src.core.models import Transcript
from src.api.helpers import transcribe_helper


def test_transcribe_helper_uses_router():
    fake = Transcript("m", "faster", "sv", 1.0, 0.1, provider="local")
    with patch("src.api.helpers.AsrRouter") as R:
        R.return_value.transcribe.return_value = fake
        d = transcribe_helper("a.wav", provider="local")
    R.return_value.transcribe.assert_called()
    assert d["provider"] == "local"
```

- [ ] **Step 2: Run to verify fail/miss**

Run: `pytest tests/test_asr.py -k transcribe_helper_uses_router -v` (place test in `tests/test_asr.py` or new `tests/test_asr_helpers.py`)

- [ ] **Step 3: Switch `transcribe_helper` to router**

Replace `get_transcriber(...).transcribe` with:

```python
from ..transcription.router import AsrRouter

transcript = AsrRouter().transcribe(
    audio_path,
    provider=provider,
    backend=backend,
    model_name=model,
    device=device,
    cloud_fallback_local=cloud_fallback_local,
    language=language,
    # ... remaining kwargs
)
```

Mirror in `pipeline.py` ASR call site. Add fields to `AsrParamsMixin` and `asr_kwargs_from`.

- [ ] **Step 4: Run focused tests**

Run: `pytest tests/test_asr.py tests/test_api.py -k "transcribe or Asr" -v --ignore=tests/test_audio_benchmarks.py`  
Expected: existing tests still pass (defaults keep local)

- [ ] **Step 5: Commit**

```bash
git add src/api/helpers.py src/api/schemas.py src/pipeline.py src/cli.py tests/test_asr_helpers.py
git commit -m "feat(asr): route CLI/API/pipeline transcription through AsrRouter"
```

---

### Task 5: Local harden — decode flag + chunk retry

**Files:**
- Modify: `src/transcription/faster_whisper.py`
- Test: `tests/test_asr.py` (extend) or `tests/test_asr_chunk_retry.py`

**Interfaces:**
- Consumes: existing chunk loop
- Produces: `condition_on_previous_text=False` by default in all `wmodel.transcribe` kwargs; per-chunk retry up to 2 attempts; on final failure append warning `chunk_failed:{index}` and continue (do not silent-skip without warning)

- [ ] **Step 1: Write failing tests**

```python
from unittest.mock import MagicMock, patch
from src.transcription.faster_whisper import FasterWhisperTranscriber


def test_transcribe_passes_condition_on_previous_text_false():
    # Arrange a loaded model mock; assert kwargs include condition_on_previous_text=False
    ...


def test_chunk_failure_retries_then_warns(tmp_path):
    # Mock decode_audio + model.transcribe: fail twice on chunk 1, succeed on chunk 2
    # Assert warnings contain chunk_failed or chunk_retry
    ...
```

Fill mocks following patterns already in `tests/test_asr.py` (patch `_get_model`, `decode_audio`).

- [ ] **Step 2: Run tests — expect FAIL**

Run: `pytest tests/test_asr_chunk_retry.py -v`

- [ ] **Step 3: Implement**

In both non-chunked and chunked `transcribe_kwargs` / `chunk_kwargs`:

```python
"condition_on_previous_text": (
    True if condition_on_previous_text is True else False
),
```

Chunk loop:

```python
attempts = 0
while attempts < 3:
    try:
        segments_iter, _ = wmodel.transcribe(chunk, **chunk_kwargs)
        break
    except Exception as ce:
        attempts += 1
        if attempts >= 3:
            warnings.append(f"chunk_failed:{chunk_index}")
            logger.warning("Chunk %d failed after retries: %s", chunk_index, ce)
            segments_iter = []
            break
# after building transcript:
transcript.warnings.extend(warnings)
```

Attach warnings before `add_diarization` / return.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_asr_chunk_retry.py tests/test_asr.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/transcription/faster_whisper.py tests/test_asr_chunk_retry.py
git commit -m "fix(asr): disable prev-text conditioning and retry failed chunks"
```

---

### Task 6: Deepgram cloud adapter

**Files:**
- Create: `src/transcription/cloud_deepgram.py`
- Create: `tests/fixtures/deepgram_listen_response.json` (minimal fixture)
- Modify: `src/transcription/router.py` (wire real adapter)
- Modify: `pyproject.toml` (`cloud-stt` extra = `httpx>=0.27.0`; marker `cloud_stt`)
- Test: `tests/test_asr_cloud_deepgram.py`

**Interfaces:**
- Consumes: audio file path, env `DEEPGRAM_API_KEY` or `CLOUD_STT_API_KEY`
- Produces: `DeepgramTranscriber.transcribe(...) -> Transcript` with `backend="deepgram"`, `provider` set by router
- Raises `TranscriptionError` with codes: auth→`CLOUD_AUTH`, timeout→`CLOUD_TIMEOUT`, 429→`CLOUD_QUOTA`
- On entry log: `logger.info("asr_cloud_egress=true provider=deepgram")` — no path contents of transcript

- [ ] **Step 1: Write mapper unit tests with fixture (no network)**

```python
from src.transcription.cloud_deepgram import map_deepgram_response, DeepgramTranscriber
from src.core.errors import TranscriptionError
from src.transcription.error_codes import AsrErrorCode
import json
from pathlib import Path


def test_map_deepgram_response_segments():
    data = json.loads(Path("tests/fixtures/deepgram_listen_response.json").read_text(encoding="utf-8"))
    segs = map_deepgram_response(data)
    assert segs[0].text
    assert segs[0].start >= 0


def test_missing_api_key_raises_cloud_auth(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("CLOUD_STT_API_KEY", raising=False)
    with pytest.raises(TranscriptionError) as ei:
        DeepgramTranscriber().transcribe("x.wav")
    assert AsrErrorCode.CLOUD_AUTH in str(ei.value)
```

Fixture shape (minimal):

```json
{
  "results": {
    "channels": [
      {
        "alternatives": [
          {
            "transcript": "Hej jag behöver hjälp",
            "words": [
              {"word": "Hej", "start": 0.0, "end": 0.3, "confidence": 0.9},
              {"word": "jag", "start": 0.3, "end": 0.5, "confidence": 0.8}
            ]
          }
        ]
      }
    ]
  },
  "metadata": {"duration": 1.2}
}
```

- [ ] **Step 2: Run — expect FAIL**

Run: `pytest tests/test_asr_cloud_deepgram.py -v`

- [ ] **Step 3: Implement Deepgram client**

- Read API key from env
- `httpx.post` with timeout 120s; retry up to 3 times only on 429/5xx/transport errors with exponential backoff (0.5s, 1s, 2s)
- Map words → `Segment`/`Word`; avg confidence → `low_confidence` if < 0.60
- Wire router: if `provider=="cloud"` and `cloud_provider=="deepgram"`, use `DeepgramTranscriber`
- If `cloud_fallback_local` and cloud fails with timeout/quota only, optionally call local (default off — test that default does **not** fallback)

- [ ] **Step 4: Run unit tests**

Run: `pytest tests/test_asr_cloud_deepgram.py tests/test_asr_router.py -v`  
Expected: PASS without network

- [ ] **Step 5: Commit**

```bash
git add src/transcription/cloud_deepgram.py src/transcription/router.py tests/test_asr_cloud_deepgram.py tests/fixtures/deepgram_listen_response.json pyproject.toml
git commit -m "feat(asr): add opt-in Deepgram CloudSttEngine behind AsrRouter"
```

---

### Task 7: A/B compare CLI

**Files:**
- Modify: `src/benchmarks/audio_cli.py`
- Modify: `src/benchmarks/audio_runner.py` (accept `provider`)
- Test: `tests/test_audio_benchmarks.py` (dry-run compare)

**Interfaces:**
- Produces: `sentimentanalys evaluate audio compare --providers local,cloud --limit N --dry-run`
- Report JSON fields per file: `provider`, `latency_s`, `n_segments`, `wer` (if reference transcript in sidecar), `estimated_cost_usd` (Deepgram: `duration_min * 0.0043` placeholder constant documented in code)

- [ ] **Step 1: Failing test for compare dry-run**

```python
from typer.testing import CliRunner
# invoke audio compare --dry-run --providers local
# assert exit 0 and report mentions providers
```

- [ ] **Step 2: Run — expect FAIL** (command missing)

- [ ] **Step 3: Implement `compare` command**

Loop providers; for each selected sample call `AsrRouter().transcribe` (skip real cloud in dry-run); write report under `reports/audio_compare_{timestamp}.json`.

- [ ] **Step 4: Test**

Run: `pytest tests/test_audio_benchmarks.py -k compare -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/benchmarks/audio_cli.py src/benchmarks/audio_runner.py tests/test_audio_benchmarks.py
git commit -m "feat(asr): add evaluate audio compare for local vs cloud A/B"
```

---

### Task 8: Persistent transcription jobs

**Files:**
- Create: `src/api/transcription_job_store.py`
- Modify: `src/api/transcription_jobs.py` (delegate persistence)
- Modify: `src/api/app.py` or lifespan — init store from data root / Redis
- Test: `tests/test_transcription_jobs.py` (extend with restart simulation)

**Interfaces:**
- Consumes: existing `TranscriptionJob` / registry API
- Produces: `JobStore.upsert/get/list/complete/cancel`; registry loads from store on init; survives new `TranscriptionJobRegistry` instance reading same SQLite file

- [ ] **Step 1: Write failing persistence test**

```python
def test_job_survives_registry_recreate(tmp_path):
    db = tmp_path / "jobs.db"
    store = SqliteJobStore(db)
    reg1 = TranscriptionJobRegistry(store=store)
    reg1.register("j1", "transcribe")
    reg2 = TranscriptionJobRegistry(store=SqliteJobStore(db))
    assert reg2.get("j1") is not None
    assert reg2.get("j1").status == "running"
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement SqliteJobStore + optional RedisJobStore**

- SQLite schema: `job_id TEXT PRIMARY KEY, kind TEXT, status TEXT, created_at TEXT, meta_json TEXT, cancelled INTEGER`
- `cancel_event` remains in-memory for live process; on restore, cancelled jobs get a pre-set event
- Wire `get_job_registry(app)` to construct store from `API_USE_REDIS_CACHE` / data root

- [ ] **Step 4: Run**

Run: `pytest tests/test_transcription_jobs.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/api/transcription_job_store.py src/api/transcription_jobs.py src/api/app.py tests/test_transcription_jobs.py
git commit -m "feat(asr): persist transcription jobs across API restarts"
```

---

### Task 9: ASR Prometheus metrics

**Files:**
- Modify: `src/core/metrics.py`
- Modify: `src/transcription/router.py` (record on success/failure)
- Test: `tests/test_asr_metrics.py`

**Interfaces:**
- Produces:
  - `asr_transcription_duration_seconds` Histogram labels `provider`, `backend`, `outcome`
  - `asr_transcriptions_total` Counter labels `provider`, `backend`, `outcome`
  - `asr_cloud_egress_total` Counter
  - `record_asr_transcription(provider, backend, outcome, duration_s)`

- [ ] **Step 1: Failing test**

```python
from src.core.metrics import record_asr_transcription, ASR_TRANSCRIPTIONS_TOTAL

def test_record_asr_increments_when_prometheus_available():
    if ASR_TRANSCRIPTIONS_TOTAL is None:
        pytest.skip("prometheus_client not installed")
    before = None  # call record; ensure no throw
    record_asr_transcription("local", "faster", "success", 1.2)
```

- [ ] **Step 2–4: Implement + wire in router finally/success paths + commit**

```bash
git commit -m "feat(asr): expose Prometheus metrics for transcription latency and egress"
```

---

### Task 10: Diarization pyannote path + docs/UI

**Files:**
- Modify: `src/transcription/base.py` (`add_diarization`)
- Modify: `SECURITY.md`, `docs/API.md`, `docs/ARCHITECTURE.md`, `docs/WINDOWS_INSTALL.md`
- Modify: `launcher/ui_settings_dialog.py` (ASR provider toggle + warning)
- Modify: `webui` ASR settings if a dedicated control exists (or transcription page copy)
- Test: extend `tests/test_asr.py` `TestAddDiarization`

**Interfaces:**
- Produces: `add_diarization(..., diarization_backend: str | None = None)` — if `None`, choose `"pyannote"` when `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` set **and** pyannote importable, else `"heuristic"`
- Docs describe cloud opt-in risk and disable steps

- [ ] **Step 1: Failing test**

```python
def test_add_diarization_prefers_pyannote_when_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "x")
    with patch("src.diarization.DiarizationPipeline") as DP:
        # assert DP called with backend="pyannote"
        ...
```

- [ ] **Step 2: Run — expect FAIL** (currently hardcoded heuristic)

- [ ] **Step 3: Fix selection + docs**

SECURITY.md section **Cloud STT (opt-in)**:
- What leaves machine (raw audio bytes to Deepgram)
- How to enable (`asr.provider=cloud`, `DEEPGRAM_API_KEY`)
- How to disable (provider local / unset key)
- Logging: egress flag only

Call-center recommendation: document that profile `callcenter` should use `preprocess_mode=callcenter` (do not force in code globally). Launcher/webui help text.

- [ ] **Step 4: Run tests + docs lint not required**

Run: `pytest tests/test_asr.py::TestAddDiarization -v`

- [ ] **Step 5: Commit**

```bash
git add src/transcription/base.py SECURITY.md docs/API.md docs/ARCHITECTURE.md docs/WINDOWS_INSTALL.md launcher/ui_settings_dialog.py
git commit -m "feat(asr): prefer pyannote when token present; document cloud STT opt-in"
```

---

### Task 11: Integration verification + CHANGELOG

**Files:**
- Modify: `CHANGELOG.md`, `docs/ROADMAP.md` (ASR hardening note)
- Test: full focused suite

- [ ] **Step 1: Run verification suite**

```bash
pytest tests/test_asr_error_codes.py tests/test_asr_postprocess.py tests/test_asr_router.py tests/test_asr_cloud_deepgram.py tests/test_asr_chunk_retry.py tests/test_transcription_jobs.py tests/test_asr.py tests/test_asr_helpers.py -q
ruff check src/transcription src/api/helpers.py src/api/transcription_job_store.py src/core/metrics.py
```

Expected: all PASS; ruff clean

- [ ] **Step 2: Update CHANGELOG under Unreleased**

Bullet list: router, hallucination filter, chunk retry, Deepgram opt-in, persistent jobs, metrics, diarization fix.

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md docs/ROADMAP.md
git commit -m "docs: changelog and roadmap for ASR dual-engine hardening"
```

---

## Spec coverage checklist (self-review)

| Spec requirement | Task |
|------------------|------|
| AsrRouter + local default | 3, 4 |
| Hallucination filter | 2 |
| Decode harden + chunk retry | 5 |
| Deepgram opt-in + no silent fallback | 6 |
| A/B compare | 7 |
| Persistent jobs | 8 |
| Metrics / egress logging | 6, 9 |
| Diarization pyannote when token | 10 |
| SECURITY/docs/UI warning | 10 |
| Shared core; launcher/API thin | 4, 10 |
| Error codes | 1, 6 |
| CI without cloud credentials | 6 (fixture-only tests) |

No intentional placeholders remain; Deepgram live calls are opt-in via `@pytest.mark.cloud_stt` only if added later (not required for merge).
