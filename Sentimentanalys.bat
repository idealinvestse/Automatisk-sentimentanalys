@echo off
REM Start Sentimentanalys GUI launcher (tkinter hub)
REM Double-click this file, or pin to Start Menu / taskbar.
setlocal EnableExtensions EnableDelayedExpansion

set "APP_ROOT=%~dp0"
cd /d "%APP_ROOT%"
set "SENTIMENT_APP_ROOT=%APP_ROOT%"
set "PYTHONPATH=%APP_ROOT%"

REM --- ffmpeg / PATH bootstrap (ASR preprocess) ---
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

set "PYW=%APP_ROOT%.venv\Scripts\pythonw.exe"
set "PY=%APP_ROOT%.venv\Scripts\python.exe"

if exist "%PYW%" (
    start "Sentimentanalys" "%PYW%" -m launcher.main
    exit /b 0
)

if exist "%PY%" (
    start "Sentimentanalys" "%PY%" -m launcher.main
    exit /b 0
)

where pythonw >nul 2>&1
if %ERRORLEVEL% equ 0 (
    start "Sentimentanalys" pythonw -m launcher.main
    exit /b 0
)

where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    start "Sentimentanalys" python -m launcher.main
    exit /b 0
)

echo.
echo  Sentimentanalys: Python hittades inte.
echo.
echo  Kor fran projektmappen (krav: Python 3.11+ pa PATH):
echo    launcher.ps1 provision
echo.
echo  Starta sedan denna fil igen, eller kor:
echo    launcher.ps1 doctor
echo.
pause
exit /b 1