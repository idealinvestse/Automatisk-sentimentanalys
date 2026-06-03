# Build full Inno Setup installer (requires Inno Setup 6 on PATH as ISCC.exe)
param(
    [string]$Version = "0.3.0",
    [ValidateSet("minimal", "cli", "api", "full")]
    [string]$Profile = "full"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

$staging = Join-Path $Root "build\portable-staging\Sentimentanalys"
if (-not (Test-Path $staging)) {
    Write-Host "==> Build portable staging first"
    & (Join-Path $PSScriptRoot "build-portable.ps1") -Version $Version -Profile $Profile
} else {
    Write-Host "==> Reusing existing staging at $staging"
}

$iscc = Get-Command ISCC.exe -ErrorAction SilentlyContinue
if (-not $iscc) {
    $defaultIscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
    if (Test-Path $defaultIscc) { $iscc = @{ Source = $defaultIscc } }
    else {
        Write-Warning "ISCC.exe not found. Portable ZIP was built; install Inno Setup 6 to compile .exe"
        exit 0
    }
}

$iss = Join-Path $PSScriptRoot "inno\sentimentanalys.iss"
& $iscc.Source $iss /DMyAppVersion=$Version /DInstallProfile=$Profile
Write-Host "==> Installer output in dist\"