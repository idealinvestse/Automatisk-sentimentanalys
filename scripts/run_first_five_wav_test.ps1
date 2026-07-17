#!/usr/bin/env pwsh
<#
.SYNOPSIS
    First-five WAV pilot helper (Windows + CUDA).

.DESCRIPTION
    Validates the sv_callcenter pack, runs ASR smoke (step A), optionally pipeline (step B),
    and prints web UI instructions (step C). Does not copy WAV files — place five demo
    .wav files under samples/audio/sv/callcenter/ before running.

.EXAMPLE
    .\scripts\run_first_five_wav_test.ps1

.EXAMPLE
    .\scripts\run_first_five_wav_test.ps1 -SkipPipeline -Device cuda -Limit 5
#>
param(
    [switch]$SkipPipeline,
    [switch]$SkipUi,
    [string]$Device = "cuda",
    [int]$Limit = 5
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$PackId = "sv_callcenter"
$ExpectedCount = 5

Set-Location $RepoRoot

function Invoke-Python {
    param([Parameter(Mandatory = $true)][string[]]$Args)
    & python @Args
    if ($LASTEXITCODE -ne 0) {
        throw "python $($Args -join ' ') failed with exit code $LASTEXITCODE"
    }
}

Write-Host "=== First five WAV test (sv_callcenter) ===" -ForegroundColor Cyan
Write-Host ""

Write-Host "[hint] Provision and doctor (run manually if not done):" -ForegroundColor Yellow
Write-Host "  .\scripts\dev-setup.ps1 -Profile cli -InitConfig" -ForegroundColor Gray
Write-Host "  .\launcher.ps1 doctor" -ForegroundColor Gray
Write-Host "  .\launcher.ps1 asr-download" -ForegroundColor Gray
Write-Host ""

Write-Host "[1/4] Validating audio catalog..." -ForegroundColor Yellow
$validateJson = & python -m src.evaluate audio validate --json
if ($LASTEXITCODE -ne 0) {
    Write-Error "audio validate failed (exit $LASTEXITCODE)"
    exit 1
}
$report = $validateJson | ConvertFrom-Json
if (-not $report.ok) {
    Write-Error "audio validate reported errors:"
    foreach ($err in $report.errors) {
        Write-Host "  - $err" -ForegroundColor Red
    }
    exit 1
}

$packInfo = $report.packs.$PackId
if (-not $packInfo) {
    Write-Error "Pack '$PackId' not found in manifest."
    exit 1
}
if (-not $packInfo.active) {
    Write-Error "Pack '$PackId' is not active. Enable it in samples/audio/manifest.yaml."
    exit 1
}

$fileCount = [int]$packInfo.file_count
Write-Host "  Pack $PackId : $fileCount file(s)" -ForegroundColor Green

if ($fileCount -eq 0) {
    Write-Error @"
No WAV files found for '$PackId'.
Place $ExpectedCount demo .wav files in samples/audio/sv/callcenter/ (gitignored) and retry.
"@
    exit 1
}
if ($fileCount -ne $ExpectedCount) {
    Write-Warning "Expected $ExpectedCount files for the first pilot, found $fileCount. Continuing."
}

Write-Host ""
Write-Host "[2/4] Listing pack samples..." -ForegroundColor Yellow
Invoke-Python -Args @(
    "-m", "src.evaluate", "audio", "list",
    "--pack", $PackId,
    "--limit", [string]([Math]::Max($Limit, 10))
)

Write-Host ""
Write-Host "[3/4] ASR smoke (step A)..." -ForegroundColor Yellow
Invoke-Python -Args @(
    "-m", "src.evaluate", "audio", "smoke",
    "--pack", $PackId,
    "--device", $Device,
    "--limit", [string]$Limit,
    "--language", "sv",
    "--oom-fallback"
)

if (-not $SkipPipeline) {
    Write-Host ""
    Write-Host "[4/4] Pipeline + LLM (step B)..." -ForegroundColor Yellow
    if (Test-Path "scripts/verify_pilot_policy.py") {
        Invoke-Python -Args @("scripts/verify_pilot_policy.py")
    } else {
        Write-Warning "scripts/verify_pilot_policy.py not found; skipping policy check."
    }
    Invoke-Python -Args @(
        "-m", "src.evaluate", "audio", "run",
        "--scenario", "pipeline",
        "--pack", $PackId,
        "--device", $Device,
        "--limit", [string]$Limit,
        "--language", "sv",
        "--oom-fallback"
    )
} else {
    Write-Host ""
    Write-Host "[4/4] Pipeline skipped (-SkipPipeline)." -ForegroundColor Gray
}

if (-not $SkipUi) {
    Write-Host ""
    Write-Host "=== Step C — API + webui ===" -ForegroundColor Cyan
    Write-Host "  .\launcher.ps1" -ForegroundColor White
    Write-Host "  Open http://localhost:3000 → Transkribering or Testlabb" -ForegroundColor White
    Write-Host "  Run one of the five demo files manually." -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "Step C instructions skipped (-SkipUi)." -ForegroundColor Gray
}

Write-Host ""
Write-Host "=== First-five WAV helper complete ===" -ForegroundColor Green
