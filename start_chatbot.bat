@echo off
title MiniChat - Local AI Server
echo ================================================
echo          MiniChat Launcher
echo ================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.10+ from python.org
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "%~dp0app.py" (
    echo ERROR: app.py not found.
    echo Please run this script from the MiniChat folder.
    pause
    exit /b 1
)

REM Navigate to script directory
cd /d "%~dp0"

REM Install dependencies if needed
echo Checking dependencies...
python -m pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies (first time only)...
    python -m pip install -r requirements.txt
    echo.
)

REM Launch Streamlit
echo Starting MiniChat server...
echo.
echo The app will open in your browser shortly.
echo To stop the server, close this window or press Ctrl+C.
echo.
python -m streamlit run "%~dp0app.py" --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false --server.address=0.0.0.0

pause
