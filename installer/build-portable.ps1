# Build portable Windows ZIP with bundled Python embed + venv + app
param(
    [string]$Version = "0.4.1",
    [ValidateSet("minimal", "cli", "api", "full")]
    [string]$Profile = "full",
    [string]$OutDir = "dist",
    [string]$PythonVersion = "3.11.9"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Staging = Join-Path $Root "build\portable-staging"
$StageApp = Join-Path $Staging "Sentimentanalys"

if (Test-Path $Staging) { Remove-Item $Staging -Recurse -Force }
New-Item -ItemType Directory -Path $StageApp -Force | Out-Null

Write-Host "==> Stage application files"
$copyItems = @("src", "app", "configs", "data", "samples", "docs", "launcher", "launcher.ps1",
    "pyproject.toml", "README.md")
foreach ($item in $copyItems) {
    $src = Join-Path $Root $item
    if (Test-Path $src) {
        Copy-Item $src (Join-Path $StageApp $item) -Recurse -Force
    }
}

New-Item -ItemType Directory -Path (Join-Path $StageApp "cache\hf") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $StageApp "cache\llm") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $StageApp "outputs") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $StageApp "user_data") -Force | Out-Null

@'
version: 1
install_profile: cli
sentiment_profile: callcenter
portable_mode: true
paths:
  app_root: .
  hf_cache: cache/hf
  outputs: outputs
'@ | Set-Content (Join-Path $StageApp "user_data\user_config.yaml") -Encoding UTF8

Write-Host "==> Download Python embed ($PythonVersion)"
$pyZip = "python-$PythonVersion-embed-amd64.zip"
$pyUrl = "https://www.python.org/ftp/python/$PythonVersion/$pyZip"
$pyDir = Join-Path $StageApp "python"
New-Item -ItemType Directory -Path $pyDir -Force | Out-Null
$zipPath = Join-Path $env:TEMP $pyZip
Invoke-WebRequest -Uri $pyUrl -OutFile $zipPath
Expand-Archive $zipPath $pyDir -Force

# Enable pip in embeddable Python
$pth = Get-ChildItem $pyDir -Filter "*._pth" | Select-Object -First 1
if ($pth) {
    (Get-Content $pth.FullName) -replace '#import site', 'import site' | Set-Content $pth.FullName
}
$embedPy = Join-Path $pyDir "python.exe"
Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile (Join-Path $pyDir "get-pip.py")
& $embedPy (Join-Path $pyDir "get-pip.py")

Write-Host "==> Create venv and install pyproject extras ($Profile)"
$venv = Join-Path $StageApp ".venv"
& $embedPy -m venv $venv
$venvPy = Join-Path $venv "Scripts\python.exe"
& $venvPy -m pip install -U pip

$extrasMap = @{
    minimal = "min,install"
    cli     = "min,cli,asr,api,dashboard-nicegui,install"
    api     = "min,asr,api,install"
    full    = "min,cli,asr,api,dashboard-nicegui,llm,training,install"
}
$extras = $extrasMap[$Profile]
& $venvPy -m pip install -e ".[$extras]"

# Launcher entry script
@'
@echo off
setlocal EnableExtensions EnableDelayedExpansion
set "APP_ROOT=%~dp0"
cd /d "%APP_ROOT%"
set "SENTIMENT_APP_ROOT=%APP_ROOT%"
set "PYTHONPATH=%APP_ROOT%"
set "FFMPEG_BIN=%APP_ROOT%tools\ffmpeg\bin"
if exist "%FFMPEG_BIN%\ffmpeg.exe" (
    set "FFMPEG_PATH=%FFMPEG_BIN%\ffmpeg.exe"
) else if not defined FFMPEG_PATH (
    where ffmpeg >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        for /f "delims=" %%F in ('where ffmpeg 2^>nul') do (
            set "FFMPEG_PATH=%%F"
            goto :ffmpeg_found
        )
    )
)
:ffmpeg_found
if defined FFMPEG_PATH (
    for %%D in ("!FFMPEG_PATH!") do set "FFMPEG_DIR=%%~dpD"
    set "PATH=!FFMPEG_DIR!;%FFMPEG_BIN%;%APP_ROOT%.venv\Scripts;%PATH%"
) else (
    set "PATH=%FFMPEG_BIN%;%APP_ROOT%.venv\Scripts;%PATH%"
)
.venv\Scripts\pythonw.exe -m launcher.main
'@ | Set-Content (Join-Path $StageApp "Sentimentanalys.bat") -Encoding ASCII

@{
    version = $Version
    profile = $Profile
    build = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
} | ConvertTo-Json | Set-Content (Join-Path $StageApp "VERSION.json")

Write-Host "==> Optional: download ffmpeg (manual step if offline)"
Write-Host "    Place ffmpeg.exe in tools\ffmpeg\bin\ for ASR preprocess"

New-Item -ItemType Directory -Path (Join-Path $Root $OutDir) -Force | Out-Null
$zipName = Join-Path $Root $OutDir "Sentimentanalys-$Version-win64-portable-$Profile.zip"
if (Test-Path $zipName) { Remove-Item $zipName -Force }
Compress-Archive -Path $StageApp -DestinationPath $zipName
Write-Host "==> Created $zipName"