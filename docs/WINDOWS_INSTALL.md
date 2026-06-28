# Windows Installation Guide

This guide covers installation on Windows 10/11.

## Recommended: Modern Python Installation (Cross-platform)

The project now uses modern Python packaging via `pyproject.toml`.

```powershell
# 1. Clone
git clone https://github.com/idealinvestse/Automatisk-sentimentanalys.git
cd Automatisk-sentimentanalys

# 2. Create virtual environment
python -m venv .venv
.\ .venv\Scripts\Activate.ps1

# 3. Install (choose profile)
pip install -U pip

# Basic CLI + ASR
pip install -e ".[cli]"

# Full call center with Speaker Diarization (recommended)
pip install -e ".[cli,diarize]"

# API + Dashboard (api extra includes core ML deps; cli adds ASR CLI tools)
pip install -e ".[cli,api]"

# Development (includes tests, linting)
pip install -e ".[dev]"
```

> **Note on Diarization**: After installing the `diarize` extra, you usually need a Hugging Face token:
> ```powershell
huggingface-cli login
```
> or set the `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` environment variable.

## Legacy / Windows-specific Methods

For users who prefer a more integrated Windows experience, the following options are still supported:

### Developer Setup (PowerShell)

```powershell
.\scripts\dev-setup.ps1 -Profile cli -InitConfig
.\launcher.ps1 doctor
```

### Portable Mode & Installer

See the original portable ZIP and Inno Setup installer instructions below (maintained for compatibility).

---

## Original Windows-specific Instructions (Legacy)

### Prerequisites

- Windows 10 or 11
- Python 3.11+ (recommended from python.org)
- Optional: NVIDIA GPU + CUDA for best ASR/diarization performance
- ffmpeg (required for `--preprocess`)

### Developer Setup (Detailed)

```powershell
# Activate environment
.\ .venv\Scripts\Activate.ps1

# Install dependencies (pyproject optional-deps)
pip install -e ".[cli,asr,install]"

# For diarization (if not using the [diarize] extra above)
pip install -e ".[diarize]"

# Optional: API
pip install -e ".[api]"
```

### Launcher & GUI

- `Sentimentanalys.bat` – Simple GUI launcher
- `python -m launcher.main` – Configuration hub
- `launcher.ps1` – Advanced CLI launcher

### API as Windows Service (NSSM)

See older instructions for running the API as a background service.

### Troubleshooting

- **Diarization fails**: Install `pyannote.audio` + login to Hugging Face (`huggingface-cli login`)
- **ffmpeg not found**: Install from https://ffmpeg.org or use the portable bundle
- **Torch / CUDA issues**: Ensure matching CUDA version or fall back to CPU
- **LLM (OpenRouter)**: Set `OPENROUTER_API_KEY` environment variable

## Environment Variables

Common variables (can be set in Windows Environment Variables or via `.env`):

- `OPENROUTER_API_KEY`
- `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`
- `SENTIMENT_API_KEY` (for API authentication)
- `API_MEDIA_ROOT`

## Next Steps

After installation, see the main [README.md](../README.md) Quickstart section and `docs/ROADMAP.md` for current capabilities.