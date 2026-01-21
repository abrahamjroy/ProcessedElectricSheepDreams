@echo off
title Processed Electric Sheep Dreams - Launcher
color 0A

echo.
echo  ===============================================
echo   PROCESSED ELECTRIC SHEEP DREAMS
echo   AI Image Generation
echo  ===============================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo.
    echo Please install Python 3.10+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [OK] Python found.

:: Check if virtual environment exists
if not exist "venv" (
    echo.
    echo [SETUP] First run detected. Setting up environment...
    echo This may take a few minutes...
    echo.
    
    python -m venv venv
    call venv\Scripts\activate.bat
    
    echo [SETUP] Installing dependencies...
    pip install -r requirements.txt --quiet
    
    echo.
    echo [SETUP] Setup complete!
) else (
    call venv\Scripts\activate.bat
)

echo.
echo [START] Launching application...
echo.
echo -----------------------------------------------
echo  Close this window to stop the application.
echo -----------------------------------------------
echo.

python app.py

pause
