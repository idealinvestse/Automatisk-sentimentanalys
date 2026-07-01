# PowerShell launcher wrapper for sentimentanalys CLI
# This script forwards all arguments to the Python CLI module

$ErrorActionPreference = "Stop"

# Try to use the virtual environment if it exists
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvActivate = Join-Path $scriptDir ".venv\Scripts\Activate.ps1"

if (Test-Path $venvActivate) {
    Write-Host "Activating virtual environment..." -ForegroundColor Cyan
    & $venvActivate
}

# Forward all arguments to Python CLI
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    $pythonCmd = Get-Command python3 -ErrorAction SilentlyContinue
}

if (-not $pythonCmd) {
    Write-Error "Python not found in PATH. Please install Python 3.11+."
    exit 1
}

# Run the CLI module with all passed arguments
& $pythonCmd -m launcher.cli $args