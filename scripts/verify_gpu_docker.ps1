#!/usr/bin/env pwsh
# Verify GPU Docker image (run on a host with NVIDIA Container Toolkit).
# See docs/PRODUCTION_CHECKLIST.md section 3.

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

Write-Host "Building GPU image..." -ForegroundColor Yellow
docker build -t sentimentanalys-gpu -f Dockerfile.gpu .

Write-Host "Checking CUDA inside container..." -ForegroundColor Yellow
docker run --gpus all --rm sentimentanalys-gpu python -c @"
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
"@

if ($LASTEXITCODE -ne 0) {
    Write-Error "GPU verification failed. Ensure NVIDIA drivers and nvidia-container-toolkit are installed."
    exit 1
}

Write-Host "GPU Docker verification passed." -ForegroundColor Green
