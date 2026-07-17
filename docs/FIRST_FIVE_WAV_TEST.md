# First five WAV test (Windows + RTX 5070)

## Prerequisites

- Python 3.11+ venv, ffmpeg, NVIDIA driver + CUDA-capable PyTorch
- Place 5 demo `.wav` in `samples/audio/sv/callcenter/` (gitignored)
- Optional: `OPENROUTER_API_KEY` in `.env` for step B
- Subagents: windows-gpu-pilot, asr-smoke-runner, pilot-policy-guard

### RTX 5070 / CUDA PyTorch (Windows)

Default `pip install torch` often gives a **CPU-only** wheel (`2.x+cpu`). For the 5070, install a CUDA build (cu128 works with driver 610+):

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade torch torchaudio --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expect `True` and `NVIDIA GeForce RTX 5070`. Note: `whisperx` may warn about torch≠2.8 — ASR smoke uses `faster-whisper` and is fine.

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

```powershell
.\launcher.ps1
```

Then open the web UI (default http://localhost:3000), go to **Transkribering** or **Testlabb**, and process one of the five demo files.

## Helper script

Runs step A (and optionally B); prints step C instructions. Does not copy WAV files — place them manually first.

```powershell
.\scripts\run_first_five_wav_test.ps1
.\scripts\run_first_five_wav_test.ps1 -SkipPipeline          # A only
.\scripts\run_first_five_wav_test.ps1 -SkipUi -Device cuda -Limit 5
```

Dry-run (no GPU required):

```powershell
python -m src.evaluate audio smoke --pack sv_callcenter --dry-run --limit 5
```
