# Post-install hook (called manually or from CI after staging)
param(
    [string]$AppDir = (Split-Path -Parent $PSScriptRoot)
)

$env:SENTIMENT_APP_ROOT = $AppDir
Set-Location $AppDir

$py = Join-Path $AppDir ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Warning "venv not found at $py"
    exit 1
}

& $py -m launcher.cli configure --init
& $py -m launcher.cli doctor
Write-Host "Post-install complete. Launch Sentimentanalys.bat or Setup Hub."