# Convenience wrapper for launcher CLI from repo root
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$Root = $PSScriptRoot
$env:SENTIMENT_APP_ROOT = $Root
Set-Location $Root

if (Test-Path ".\.venv\Scripts\python.exe") {
    & .\.venv\Scripts\python.exe -m launcher.cli @Args
} else {
    python -m launcher.cli @Args
}