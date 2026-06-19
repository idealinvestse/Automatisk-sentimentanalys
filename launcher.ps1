# Convenience wrapper for launcher CLI from repo root
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$env:SENTIMENT_APP_ROOT = $Root
$env:PYTHONPATH = $Root
Set-Location $Root

# ffmpeg bootstrap (ASR preprocess)
$BundledFfmpeg = Join-Path $Root "tools\ffmpeg\bin\ffmpeg.exe"
if (Test-Path $BundledFfmpeg) {
    $env:FFMPEG_PATH = $BundledFfmpeg
} elseif (-not $env:FFMPEG_PATH) {
    $ffmpegCmd = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($ffmpegCmd) {
        $env:FFMPEG_PATH = $ffmpegCmd.Source
    }
}

$pathParts = @()
if ($env:FFMPEG_PATH) {
    $pathParts += (Split-Path $env:FFMPEG_PATH -Parent)
}
$pathParts += (Join-Path $Root "tools\ffmpeg\bin")
$venvScripts = Join-Path $Root ".venv\Scripts"
if (Test-Path $venvScripts) {
    $pathParts += $venvScripts
}
$env:PATH = ($pathParts + $env:PATH) -join ";"

if (Test-Path ".\.venv\Scripts\python.exe") {
    & .\.venv\Scripts\python.exe -m launcher.cli @Args
} else {
    python -m launcher.cli @Args
}