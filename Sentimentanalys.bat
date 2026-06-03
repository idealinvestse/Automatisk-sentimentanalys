@echo off
REM Start Sentimentanalys GUI launcher (tkinter hub)
REM Double-click this file, or pin to Start Menu / taskbar.
setlocal EnableExtensions

set "APP_ROOT=%~dp0"
cd /d "%APP_ROOT%"
set "SENTIMENT_APP_ROOT=%APP_ROOT%"

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
echo  Kor kor fran projektmappen:
echo    scripts\dev-setup.ps1 -Profile cli -InitConfig
echo.
echo  Starta sedan denna fil igen, eller kor:
echo    launcher.ps1 doctor
echo.
pause
exit /b 1