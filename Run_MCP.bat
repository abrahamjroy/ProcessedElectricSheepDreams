@echo off
title Z-Image-Turbo MCP Server
color 0B

echo.
echo  ===============================================
echo   Z-IMAGE-TURBO MCP SERVER
echo   Connect via stdio
echo  ===============================================
echo.

:: Check for venv
if not exist "venv" (
    echo [ERROR] Virtual environment not found. 
    echo Please run Launch.bat first to set up the environment.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo [INFO] MCP Server Ready.
echo [INFO] Listening on stdio...
echo.

python mcp_server.py
