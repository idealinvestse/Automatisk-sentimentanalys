# Developer setup for Automatisk-sentimentanalys (Windows PowerShell)
param(
    [ValidateSet("minimal", "cli", "api", "full", "dev")]
    [string]$Profile = "cli",
    [switch]$Cuda,
    [switch]$InitConfig
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "==> Dev setup in $Root (profile: $Profile)"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python 3.11+ not found on PATH. Install from https://www.python.org/downloads/"
}

$pyVer = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ([version]$pyVer -lt [version]"3.11") {
    throw "Python 3.11+ required, found $pyVer"
}

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}
& .\.venv\Scripts\Activate.ps1
python -m pip install -U pip

$extrasMap = @{
    minimal = "min,install"
    cli     = "min,cli,asr,api,dashboard-nicegui,install"
    api     = "min,asr,api,install"
    full    = "min,cli,asr,api,dashboard-nicegui,llm,training,install"
    dev     = "min,cli,asr,api,dashboard-nicegui,llm,training,install,dev,diarize"
}

$extras = $extrasMap[$Profile]
Write-Host "==> pip install -e .[$extras]"
pip install -e ".[$extras]"

if ($Cuda) {
    Write-Host "==> Installing CUDA torch wheel"
    pip install torch --index-url https://download.pytorch.org/whl/cu124
}

$env:SENTIMENT_APP_ROOT = $Root
if ($InitConfig) {
    python -m launcher.cli configure --init
    Write-Host "==> Created user_config.yaml"
}

Write-Host ""
Write-Host "Done. Next steps:"
Write-Host "  .\launcher.ps1 doctor"
Write-Host "  python -m launcher.main          # GUI launcher"
Write-Host "  python -m launcher.cli start-api"
Write-Host "  python -m app.nicegui_dashboard.main"
