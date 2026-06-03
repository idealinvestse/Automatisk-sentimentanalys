# Windows installer build (maintainer)

## Prerequisites

- Windows 10/11 x64
- PowerShell 5.1+
- Network access (downloads Python embed + pip packages)
- Optional: [Inno Setup 6](https://jrsoftware.org/isinfo.php) for `.exe` installer

## Quick start (developers)

From repo root:

```powershell
.\scripts\dev-setup.ps1 -Profile cli -InitConfig
.\launcher.ps1 doctor
python -m launcher.main
```

## Build portable ZIP

```powershell
.\installer\build-portable.ps1 -Version 0.3.0 -Profile full
```

Output: `dist\Sentimentanalys-0.3.0-win64-portable-full.zip`

Place `ffmpeg.exe` in `tools\ffmpeg\bin\` inside the staging tree before zipping if you need ASR preprocess (or install ffmpeg system-wide).

## Build Inno Setup installer

1. Run portable staging (build-installer does this automatically):

```powershell
.\installer\build-installer.ps1 -Version 0.3.0 -Profile full
```

2. Output: `dist\Sentimentanalys-Setup-0.3.0-full.exe`

## CI

Tag push `v*` triggers [`.github/workflows/windows-installer.yml`](../.github/workflows/windows-installer.yml).

Manual run: Actions → Windows Installer → choose profile.

## Layout (installed)

See [docs/WINDOWS_INSTALL.md](../docs/WINDOWS_INSTALL.md).