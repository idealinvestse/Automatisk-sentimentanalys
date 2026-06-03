# Windows installation guide

Three ways to run **Automatisk sentimentanalys** on Windows:

| Tier | Audience | How |
|------|----------|-----|
| **Dev** | Developers (git clone) | [`scripts/dev-setup.ps1`](../scripts/dev-setup.ps1) |
| **Portable** | Power users, no admin | ZIP from releases or `installer/build-portable.ps1` |
| **Installer** | Analysts / IT | `Sentimentanalys-Setup-*.exe` (Inno Setup) |

## Developer setup

```powershell
git clone https://github.com/ideal-invest/Automatisk-sentimentanalys.git
cd Automatisk-sentimentanalys
.\scripts\dev-setup.ps1 -Profile full -InitConfig
.\launcher.ps1 doctor
```

Profiles: `minimal`, `cli`, `api`, `full`, `dev`.

Optional CUDA torch:

```powershell
.\scripts\dev-setup.ps1 -Profile full -Cuda
```

## Launcher

After setup, double-click **`Sentimentanalys.bat`** in the project or install folder (starts the GUI without a console window when `.venv` exists).

Or from PowerShell:

```powershell
# GUI hub (tkinter)
python -m launcher.main

# CLI (IT automation)
.\launcher.ps1 doctor
.\launcher.ps1 start-api
.\launcher.ps1 start-dashboard
.\launcher.ps1 configure --init
.\launcher.ps1 set-secret openrouter --from-file configs\openrouter.key
```

Setup Hub (Streamlit):

```powershell
streamlit run app/setup_hub.py
```

## Configuration

User settings: `%AppData%\Sentimentanalys\user_config.yaml`

Portable mode: `{install}\user_data\user_config.yaml`

Shipped defaults: [`configs/install_defaults.yaml`](../configs/install_defaults.yaml)

**Secrets (recommended):** Windows Credential Manager via Setup Hub or:

```powershell
.\launcher.ps1 set-secret openrouter --value "sk-or-..."
.\launcher.ps1 set-secret huggingface --value "hf_..."
```

Environment variables still work: `OPENROUTER_API_KEY`, `HF_TOKEN`.

## Portable ZIP

1. Extract ZIP to e.g. `C:\Tools\Sentimentanalys`
2. Run `Sentimentanalys.bat` (GUI launcher)
3. First run: Setup Hub → Save → Doctor

Bundled: embedded Python, `.venv`, app code. Models download on first use (HF cache under `cache/hf`).

## Full installer

1. Run `Sentimentanalys-Setup-*.exe`
2. Choose install profile (minimal / cli / api / full)
3. Finish wizard → optional Setup Hub
4. Start Menu → **Sentimentanalys Launcher**

Uninstall removes the program folder by default. User data under `%AppData%\Sentimentanalys` is only removed if you select **Ta bort användardata** during uninstall. API keys in Windows Credential Manager are not removed automatically.

## API as a service (IT)

Optional NSSM (not bundled by default):

```powershell
nssm install SentimentanalysAPI "C:\Program Files\Sentimentanalys\.venv\Scripts\uvicorn.exe" "src.api:app" --host 0.0.0.0 --port 8000
```

Or use `.\launcher.ps1 start-api` for interactive sessions.

## Requirements

- Windows 10/11 x64
- ~5–15 GB disk (full profile + models)
- NVIDIA driver optional (CUDA ASR/sentiment)
- **ffmpeg** for `--preprocess` (bundled path `tools\ffmpeg\bin` or system PATH)

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Doctor: ffmpeg FAIL | Install ffmpeg or add to `tools\ffmpeg\bin` |
| Doctor: torch missing | `.\launcher.ps1 repair --profile cli` |
| LLM disabled | Set OpenRouter key in Setup Hub → Secrets |
| pyannote diarization | Set `HF_TOKEN` in Secrets |
| API port in use | Setup Hub → change API port → Save |

## Build from source

See [`installer/README.md`](../installer/README.md).