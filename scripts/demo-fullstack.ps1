#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Full-stack demo script for Swedish Sentiment Analysis

.DESCRIPTION
    Starts the backend API and Next.js webui, then runs a smoke test.
    This script demonstrates the complete E2E flow from transcription to dashboard.

.EXAMPLE
    .\scripts\demo-fullstack.ps1

.NOTES
    Requires Python 3.11+ and Node.js 20+ to be installed.
    Backend runs on http://localhost:8000
    WebUI runs on http://localhost:3000
#>

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

Write-Host "=== Swedish Sentiment Analysis - Full Stack Demo ===" -ForegroundColor Cyan
Write-Host ""

# Check prerequisites
Write-Host "[1/5] Checking prerequisites..." -ForegroundColor Yellow

if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python not found. Please install Python 3.11+ and add to PATH."
    exit 1
}

if (!(Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Error "Node.js not found. Please install Node.js 20+ and add to PATH."
    exit 1
}

$PythonVersion = python --version
$NodeVersion = node --version
Write-Host "  Python: $PythonVersion" -ForegroundColor Green
Write-Host "  Node.js: $NodeVersion" -ForegroundColor Green

# Install backend dependencies
Write-Host ""
Write-Host "[2/5] Installing backend dependencies..." -ForegroundColor Yellow
Set-Location $RepoRoot
pip install -e ".[api]" --quiet
Write-Host "  Backend dependencies installed" -ForegroundColor Green

# Install frontend dependencies
Write-Host ""
Write-Host "[3/5] Installing frontend dependencies..." -ForegroundColor Yellow
Set-Location "$RepoRoot\webui"
npm install --quiet
Write-Host "  Frontend dependencies installed" -ForegroundColor Green

# Start backend API
Write-Host ""
Write-Host "[4/5] Starting backend API (http://localhost:8000)..." -ForegroundColor Yellow
Set-Location $RepoRoot
$BackendJob = Start-Job -ScriptBlock {
    Set-Location $using:RepoRoot
    uvicorn src.api:app --host 0.0.0.0 --port 8000
}

# Wait for backend to be ready
Write-Host "  Waiting for backend to start..." -ForegroundColor Gray
$BackendReady = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        $Response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 2
        if ($Response.StatusCode -eq 200) {
            $BackendReady = $true
            Write-Host "  Backend is ready!" -ForegroundColor Green
            break
        }
    } catch {
        Start-Sleep -Seconds 1
    }
}

if (-not $BackendReady) {
    Write-Error "Backend failed to start. Check logs above."
    Stop-Job $BackendJob
    exit 1
}

# Start webui
Write-Host ""
Write-Host "Starting Next.js webui (http://localhost:3000)..." -ForegroundColor Yellow
Set-Location "$RepoRoot\webui"
$WebuiJob = Start-Job -ScriptBlock {
    Set-Location $using:RepoRoot\webui
    npm run dev
}

# Wait for webui to be ready
Write-Host "  Waiting for webui to start..." -ForegroundColor Gray
$WebuiReady = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        $Response = Invoke-WebRequest -Uri "http://localhost:3000" -UseBasicParsing -TimeoutSec 2
        if ($Response.StatusCode -eq 200) {
            $WebuiReady = $true
            Write-Host "  WebUI is ready!" -ForegroundColor Green
            break
        }
    } catch {
        Start-Sleep -Seconds 1
    }
}

if (-not $WebuiReady) {
    Write-Error "WebUI failed to start. Check logs above."
    Stop-Job $BackendJob
    Stop-Job $WebuiJob
    exit 1
}

# Run smoke test
Write-Host ""
Write-Host "[5/5] Running smoke test..." -ForegroundColor Yellow
Set-Location "$RepoRoot\webui"
npm run test:e2e

Write-Host ""
Write-Host "Docker staging (optional production-like stack):" -ForegroundColor Cyan
Write-Host "  docker compose -f docker-compose.staging.yml up --build" -ForegroundColor White
Write-Host "  python scripts/staging_observability_smoke.py --api-key staging-local-dev-key" -ForegroundColor Gray

# Summary
Write-Host ""
Write-Host "=== Demo Complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Services are running:" -ForegroundColor Green
Write-Host "  Backend API:  http://localhost:8000" -ForegroundColor White
Write-Host "  Next.js WebUI: http://localhost:3000" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop all services." -ForegroundColor Yellow

# Keep script running until Ctrl+C
try {
    while ($true) {
        Start-Sleep -Seconds 1
    }
} finally {
    Write-Host ""
    Write-Host "Stopping services..." -ForegroundColor Yellow
    Stop-Job $BackendJob
    Stop-Job $WebuiJob
    Remove-Job $BackendJob
    Remove-Job $WebuiJob
    Write-Host "All services stopped." -ForegroundColor Green
}
