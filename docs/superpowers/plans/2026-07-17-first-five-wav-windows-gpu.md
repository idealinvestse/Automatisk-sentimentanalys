# First Five WAV Windows GPU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Windows + RTX 5070 host ready to run five Swedish demo WAVs through A (CUDA ASR smoke) → B (pipeline + OpenRouter) → C (API + webui), with automatic `kb-whisper-large` → `kb-whisper-medium` on CUDA OOM.

**Architecture:** Add a small pure helper for OOM-aware local transcription; thread `model_name` + `oom_fallback` through `audio_runner` / evaluate CLI; fix Swedish pack smoke selection so non-RAVDESS packs can select all five files; ship an operator runbook. Subagents already live under `.cursor/agents/`.

**Tech Stack:** Python 3.11+, faster-whisper / KB-Whisper, Typer evaluate CLI, pytest, PowerShell launcher, OpenRouter LLM.

**Spec:** `docs/superpowers/specs/2026-07-17-first-five-wav-windows-gpu-design.md`

## Global Constraints

- Native Windows + launcher only (no Docker GPU in this plan)
- ASR `provider=local` only; never enable Deepgram/Groq for this path
- Demo WAVs under `samples/audio/sv/callcenter/`; must stay gitignored (`samples/audio/**/*.wav` already in `.gitignore`)
- OOM: one retry to `kb-whisper-medium`; no silent CPU fallback in GPU-test mode (`device=cuda`)
- LLM: OpenRouter present for step B; callcenter anonymize before LLM
- TDD: failing test before implementation in each code task
- Conventional commits (`feat:`, `fix:`, `docs:`, `test:`)
- Never commit `.wav`, `.env`, or API keys

## File Structure (locked)

| Path | Responsibility |
|------|----------------|
| `src/transcription/base.py` | Add `kb-whisper-medium` alias |
| `src/transcription/oom_fallback.py` | Detect CUDA OOM; retry with fallback model |
| `src/benchmarks/audio_runner.py` | Pass `model_name`; use OOM helper; GPU-strict device |
| `src/benchmarks/audio_cli.py` | `--model`, `--oom-fallback`, `--limit` on smoke |
| `src/benchmarks/audio_catalog.py` | Non-RAVDESS `smoke_subset` → first N samples |
| `src/benchmarks/audio_scenarios.py` | Optional: document limit overrides for five-file pack |
| `samples/audio/manifest.yaml` | `sv_callcenter.enabled: true` |
| `samples/audio/sv/callcenter/.gitkeep` | Ensure directory exists without audio |
| `docs/FIRST_FIVE_WAV_TEST.md` | Operator A→B→C runbook |
| `scripts/run_first_five_wav_test.ps1` | Host checklist / smoke launcher |
| `.cursor/agents/*.md` | Already created — verify only |
| `tests/test_asr_oom_fallback.py` | Unit tests for OOM helper |
| `tests/test_audio_pack_sv_callcenter.py` | Pack/subset selection tests |

---

### Task 1: `kb-whisper-medium` alias

**Files:**
- Modify: `src/transcription/base.py` (`_MODEL_ALIASES`)
- Test: `tests/test_asr_helpers.py` (extend) or create assertions in existing helper tests

**Interfaces:**
- Consumes: `resolve_model_name(name: str) -> str`
- Produces: `resolve_model_name("kb-whisper-medium") == "KBLab/kb-whisper-medium"`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_asr_helpers.py (or create if missing)
from src.transcription.base import resolve_model_name


def test_kb_whisper_medium_alias():
    assert resolve_model_name("kb-whisper-medium") == "KBLab/kb-whisper-medium"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_asr_helpers.py::test_kb_whisper_medium_alias -v`  
Expected: FAIL (alias missing or wrong)

- [ ] **Step 3: Minimal implementation**

In `src/transcription/base.py` `_MODEL_ALIASES`:

```python
_MODEL_ALIASES = {
    "kb-whisper-large": "KBLab/kb-whisper-large",
    "kb-whisper-medium": "KBLab/kb-whisper-medium",
    "large-v3": "openai/whisper-large-v3",
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_asr_helpers.py::test_kb_whisper_medium_alias -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/transcription/base.py tests/test_asr_helpers.py
git commit -m "feat(asr): add kb-whisper-medium model alias"
```

---

### Task 2: CUDA OOM fallback helper

**Files:**
- Create: `src/transcription/oom_fallback.py`
- Test: `tests/test_asr_oom_fallback.py`

**Interfaces:**
- Consumes: callable `transcribe_fn(model_name: str) -> T`
- Produces:

```python
@dataclass(frozen=True)
class OomFallbackResult:
    value: object
    model_used: str
    fell_back: bool


def is_cuda_oom_error(exc: BaseException) -> bool: ...


def transcribe_with_oom_fallback(
    *,
    primary_model: str = "kb-whisper-large",
    fallback_model: str = "kb-whisper-medium",
    allow_fallback: bool = True,
    on_fallback: callable | None = None,
    transcribe_fn: callable,
) -> OomFallbackResult: ...
```

Rules:
- Call `transcribe_fn(primary_model)` first
- On CUDA OOM only (message/type match): if `allow_fallback` and models differ, call `on_fallback` then `transcribe_fn(fallback_model)` once
- Non-OOM errors re-raise
- Second OOM re-raises
- Never catch-all Exception for non-OOM

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_asr_oom_fallback.py
import pytest

from src.transcription.oom_fallback import (
    OomFallbackResult,
    is_cuda_oom_error,
    transcribe_with_oom_fallback,
)


class FakeCudaOom(RuntimeError):
    def __init__(self) -> None:
        super().__init__("CUDA out of memory. Tried to allocate 512 MiB")


def test_is_cuda_oom_detects_message():
    assert is_cuda_oom_error(FakeCudaOom()) is True
    assert is_cuda_oom_error(ValueError("bad audio")) is False


def test_fallback_on_oom():
    calls: list[str] = []

    def fn(model: str) -> str:
        calls.append(model)
        if model == "kb-whisper-large":
            raise FakeCudaOom()
        return "ok"

    result = transcribe_with_oom_fallback(transcribe_fn=fn)
    assert isinstance(result, OomFallbackResult)
    assert result.value == "ok"
    assert result.model_used == "kb-whisper-medium"
    assert result.fell_back is True
    assert calls == ["kb-whisper-large", "kb-whisper-medium"]


def test_no_fallback_when_disabled():
    def fn(model: str) -> str:
        raise FakeCudaOom()

    with pytest.raises(RuntimeError, match="out of memory"):
        transcribe_with_oom_fallback(transcribe_fn=fn, allow_fallback=False)


def test_non_oom_propagates():
    def fn(model: str) -> str:
        raise ValueError("corrupt wav")

    with pytest.raises(ValueError, match="corrupt wav"):
        transcribe_with_oom_fallback(transcribe_fn=fn)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_asr_oom_fallback.py -v`  
Expected: FAIL (module missing)

- [ ] **Step 3: Minimal implementation**

```python
# src/transcription/oom_fallback.py
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")

_OOM_MARKERS = (
    "out of memory",
    "cuda out of memory",
    "cudnn_status_alloc_failed",
    "hip out of memory",
)


@dataclass(frozen=True)
class OomFallbackResult:
    value: object
    model_used: str
    fell_back: bool


def is_cuda_oom_error(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    if "outofmemory" in name:
        return True
    msg = str(exc).lower()
    return any(marker in msg for marker in _OOM_MARKERS)


def transcribe_with_oom_fallback(
    *,
    primary_model: str = "kb-whisper-large",
    fallback_model: str = "kb-whisper-medium",
    allow_fallback: bool = True,
    on_fallback: Callable[[str, str, BaseException], None] | None = None,
    transcribe_fn: Callable[[str], T],
) -> OomFallbackResult:
    try:
        value = transcribe_fn(primary_model)
        return OomFallbackResult(value=value, model_used=primary_model, fell_back=False)
    except Exception as exc:
        if not allow_fallback or not is_cuda_oom_error(exc):
            raise
        if fallback_model == primary_model:
            raise
        if on_fallback is not None:
            on_fallback(primary_model, fallback_model, exc)
        else:
            logger.warning(
                "CUDA OOM with model %s; retrying once with %s",
                primary_model,
                fallback_model,
            )
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        value = transcribe_fn(fallback_model)
        return OomFallbackResult(value=value, model_used=fallback_model, fell_back=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_asr_oom_fallback.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/transcription/oom_fallback.py tests/test_asr_oom_fallback.py
git commit -m "feat(asr): add CUDA OOM large-to-medium fallback helper"
```

---

### Task 3: Wire model + OOM into audio benchmarks

**Files:**
- Modify: `src/benchmarks/audio_runner.py` (`_run_asr_on_sample`, `run_scenario` signature)
- Modify: `src/benchmarks/audio_cli.py` (`smoke`, `run` options)
- Test: `tests/test_audio_benchmarks.py` (extend mocked ASR test)

**Interfaces:**
- Consumes: `transcribe_with_oom_fallback`, `AsrRouter.transcribe`
- Produces: `run_scenario(..., model_name: str = "kb-whisper-large", oom_fallback: bool = True)`
- CLI: `--model`, `--oom-fallback/--no-oom-fallback`, smoke gains `--limit`

Behavior:
- `_run_asr_on_sample` wraps router call with OOM helper when `oom_fallback` is True
- When `device == "cuda"` and CUDA unavailable: raise clear `RuntimeError` before ASR (no silent CPU)
- Include `model_used` / `fell_back` in file result metadata or summary counters (`oom_fallbacks`)

- [ ] **Step 1: Write failing test (mocked OOM then success)**

```python
# tests/test_audio_benchmarks.py — add
from unittest.mock import patch

from src.core.models import Segment, Transcript


@patch("src.benchmarks.audio_runner.scenario_requires_ml", return_value=False)
@patch("src.transcription.router.AsrRouter.transcribe")
def test_smoke_oom_fallback_uses_medium(mock_transcribe, _mock_requires_ml, tmp_path):
    audio_root, _ = _audio_root(tmp_path)
    calls: list[str] = []

    def _side_effect(*args, **kwargs):
        model = kwargs.get("model_name", "kb-whisper-large")
        calls.append(model)
        if model == "kb-whisper-large":
            raise RuntimeError("CUDA out of memory")
        return Transcript(
            model=model,
            backend="faster",
            language="sv",
            duration=1.0,
            processing_time=0.1,
            segments=[Segment(start=0.0, end=1.0, text="hej")],
        )

    mock_transcribe.side_effect = _side_effect
    report = run_scenario(
        "smoke",
        audio_root=audio_root,
        device="cpu",  # unit test host; OOM path still exercised via mock
        model_name="kb-whisper-large",
        oom_fallback=True,
    )
    assert report.summary.get("oom_fallbacks", 0) >= 1
    assert "kb-whisper-medium" in calls
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_audio_benchmarks.py::test_smoke_oom_fallback_uses_medium -v`  
Expected: FAIL (`model_name` unexpected kwarg or no fallback)

- [ ] **Step 3: Implement wiring**

In `audio_runner.py`:

```python
def _run_asr_on_sample(
    sample_path: str,
    *,
    backend: str,
    device: str,
    language: str,
    provider: str = "local",
    model_name: str = "kb-whisper-large",
    oom_fallback: bool = True,
) -> tuple[str, float, str, bool]:
    from ..transcription.oom_fallback import transcribe_with_oom_fallback
    from ..transcription.router import AsrRouter

    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "device=cuda requested but torch.cuda.is_available() is False"
            )

    start = time.time()
    router = AsrRouter()

    def _call(model: str):
        return router.transcribe(
            sample_path,
            provider=provider,
            backend=backend,
            device=device,
            language=language,
            model_name=model,
        )

    result = transcribe_with_oom_fallback(
        primary_model=model_name,
        fallback_model="kb-whisper-medium",
        allow_fallback=oom_fallback,
        transcribe_fn=_call,
    )
    elapsed = time.time() - start
    return _transcript_text(result.value), elapsed, result.model_used, result.fell_back
```

Update all call sites in `run_scenario` / compare path to unpack the new return and increment `oom_fallbacks`.

In `audio_cli.py` add to `audio_smoke` and `audio_run`:

```python
model: str = typer.Option("kb-whisper-large", "--model"),
oom_fallback: bool = typer.Option(True, "--oom-fallback/--no-oom-fallback"),
# smoke only:
limit: int | None = typer.Option(None, "--limit"),
```

Pass through to `run_scenario`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_audio_benchmarks.py::test_smoke_oom_fallback_uses_medium tests/test_audio_benchmarks.py::test_smoke_with_mocked_asr -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/benchmarks/audio_runner.py src/benchmarks/audio_cli.py tests/test_audio_benchmarks.py
git commit -m "feat(eval): wire ASR model and CUDA OOM fallback into audio smoke"
```

---

### Task 4: Swedish pack selection for five files

**Files:**
- Modify: `samples/audio/manifest.yaml` (`sv_callcenter.enabled: true`)
- Modify: `src/benchmarks/audio_catalog.py` (`_apply_subset` for `smoke_subset`)
- Create: `samples/audio/sv/callcenter/.gitkeep`
- Create: `samples/audio/sv/callcenter/EXAMPLE.meta.yaml` (template only — no wav)
- Test: `tests/test_audio_pack_sv_callcenter.py`

**Interfaces:**
- When pack has no RAVDESS emotion codes, `smoke_subset` returns `samples[:3]` today — change to: if no RAVDESS matches, return all samples up to the effective limit (caller passes limit).
- Preferred CLI for five files after change:

```powershell
python -m src.evaluate audio smoke --pack sv_callcenter --device cuda --limit 5 --language sv
python -m src.evaluate audio run --scenario pipeline --pack sv_callcenter --device cuda --limit 5 --language sv
```

- [ ] **Step 1: Write failing test**

```python
# tests/test_audio_pack_sv_callcenter.py
from pathlib import Path

from src.benchmarks.audio_catalog import AudioCatalog
from src.benchmarks.audio_models import SampleFilter


def test_smoke_subset_returns_sidecar_pack_files(tmp_path: Path):
    root = tmp_path / "audio"
    pack = root / "sv" / "callcenter"
    pack.mkdir(parents=True)
    for i in range(5):
        (pack / f"demo_{i}.wav").write_bytes(b"RIFF")  # placeholder bytes for discovery
    (root / "manifest.yaml").write_text(
        """
version: 1
packs:
  sv_callcenter:
    label: test
    language: sv
    root: sv/callcenter
    glob: "**/*.wav"
    parser: sidecar
    default_asr_language: sv
    tags: [swedish]
    enabled: true
""",
        encoding="utf-8",
    )
    catalog = AudioCatalog(root)
    samples = catalog.discover(
        SampleFilter(pack_ids=["sv_callcenter"], subset="smoke_subset", limit=5)
    )
    assert len(samples) == 5
```

Note: if discovery requires valid wav headers beyond `RIFF`, use the same minimal fixture pattern as `tests/fixtures/` (copy approach from existing audio fixtures). Adjust fixture bytes to whatever `discover` already accepts in-repo.

- [ ] **Step 2: Run test to verify fail/skip reason, then fix subset**

In `_apply_subset` for `smoke_subset`, after the RAVDESS loop:

```python
        if chosen:
            return chosen
        # Non-RAVDESS packs (e.g. sv_callcenter): take first N in discover order
        return samples[:3] if not samples else samples[: min(3, len(samples))]
```

Change to honor that when RAVDESS selection is empty, return `samples` unchanged (limit already applied by `SampleFilter` / discover). Exact preferred logic:

```python
        if chosen:
            return chosen
        return samples  # limit already applied upstream when provided
```

Ensure `resolve_samples` applies `limit` after subset OR discover applies limit — read current order and keep five-file CLI working. If limit is applied before subset and subset truncates to 3, fix order so `--limit 5` wins for non-RAVDESS.

- [ ] **Step 3: Enable pack + template**

`samples/audio/manifest.yaml`: set `sv_callcenter.enabled: true`.

`samples/audio/sv/callcenter/EXAMPLE.meta.yaml`:

```yaml
expected_sentiment: negativ
scenario: billing_complaint
speakers: 2
notes: "Demo template — copy to <audio_stem>.meta.yaml; no real customer PII"
```

- [ ] **Step 4: Run unit test**

Run: `pytest tests/test_audio_pack_sv_callcenter.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add samples/audio/manifest.yaml samples/audio/sv/callcenter/.gitkeep samples/audio/sv/callcenter/EXAMPLE.meta.yaml src/benchmarks/audio_catalog.py tests/test_audio_pack_sv_callcenter.py
git commit -m "feat(eval): enable sv_callcenter pack and smoke selection for five WAVs"
```

---

### Task 5: Operator runbook + PowerShell helper

**Files:**
- Create: `docs/FIRST_FIVE_WAV_TEST.md`
- Create: `scripts/run_first_five_wav_test.ps1`
- Modify: `README.md` — one link under Snabbstart / Windows
- Verify: `.cursor/agents/windows-gpu-pilot.md`, `asr-smoke-runner.md`, `pilot-policy-guard.md` exist (no content rewrite unless path drift)

**Interfaces:**
- Script parameters: `-SkipPipeline`, `-SkipUi`, `-Device cuda`, `-Limit 5`
- Script steps: doctor hint → list/validate → smoke → optional pipeline → print UI next steps
- Does **not** copy WAV files (user places them)

- [ ] **Step 1: Write runbook** with exact commands:

```markdown
# First five WAV test (Windows + RTX 5070)

## Prerequisites
- Python 3.11+ venv, ffmpeg, NVIDIA driver + CUDA-capable PyTorch
- Place 5 demo `.wav` in `samples/audio/sv/callcenter/` (gitignored)
- Optional: `OPENROUTER_API_KEY` in `.env` for step B
- Subagents: windows-gpu-pilot, asr-smoke-runner, pilot-policy-guard

## A — ASR smoke
```powershell
.\scripts\dev-setup.ps1 -Profile cli -InitConfig
.\launcher.ps1 doctor
.\launcher.ps1 asr-download
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
python -m src.evaluate audio validate
python -m src.evaluate audio list --pack sv_callcenter --limit 10
python -m src.evaluate audio smoke --pack sv_callcenter --device cuda --limit 5 --language sv --oom-fallback
```

## B — Pipeline + LLM
```powershell
python scripts/verify_pilot_policy.py
python -m src.evaluate audio run --scenario pipeline --pack sv_callcenter --device cuda --limit 5 --language sv --oom-fallback
```

## C — API + webui
Start via launcher; open Transkribering / Testlabb; run one of the five files.
```

- [ ] **Step 2: Write `scripts/run_first_five_wav_test.ps1`** that runs validate + list + smoke (A), and optionally B; prints C instructions. Exit non-zero if validate fails or file count ≠ 5 (warn if ≠ 5, fail if 0).

- [ ] **Step 3: Link from README**

- [ ] **Step 4: Manual dry-run on host (no GPU required for dry-run)**

```powershell
python -m src.evaluate audio smoke --pack sv_callcenter --dry-run --limit 5
```

Expected: selects up to 5 files when WAVs present; otherwise documents “place files”.

- [ ] **Step 5: Commit**

```bash
git add docs/FIRST_FIVE_WAV_TEST.md scripts/run_first_five_wav_test.ps1 README.md .cursor/agents/
git commit -m "docs: add first-five WAV Windows GPU runbook and helper script"
```

---

### Task 6: Host verification gate (manual on RTX 5070)

**Files:** none (execution only)

**Interfaces:** Uses deliverables from Tasks 1–5

- [ ] **Step 1:** Place 5 demo WAVs in `samples/audio/sv/callcenter/`; confirm `git status` does not list them as new tracked files (`git check-ignore -v path\to\file.wav`).

- [ ] **Step 2:** Run A via script or commands in runbook; record model used (large vs medium).

- [ ] **Step 3:** Run B with OpenRouter; confirm no cloud STT.

- [ ] **Step 4:** Run C — API health + one UI transcription.

- [ ] **Step 5:** Fill success board from design §3 into chat/PR description (no commit of reports with PII).

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| Native Windows + launcher | Task 5–6 |
| 5 WAV under sv/callcenter + gitignore | Task 4–6 |
| Enable `sv_callcenter` | Task 4 |
| A→B→C order | Task 5 runbook, Task 6 |
| large → medium OOM | Tasks 1–3 |
| OpenRouter for B | Task 5–6 |
| No Docker GPU / no cloud STT | Global + runbook + policy subagent |
| Subagents | Created; Task 5 verify |
| Runbook `docs/FIRST_FIVE_WAV_TEST.md` | Task 5 |

## Self-review notes

- No TBD placeholders
- `OomFallbackResult` / `transcribe_with_oom_fallback` names consistent across Tasks 2–3
- smoke default limit remains 3 for RAVDESS; five-file path uses `--limit 5`
- WAV binaries are never committed; only `.gitkeep` + EXAMPLE meta template
