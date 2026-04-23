@echo off
chcp 65001 >nul
echo ========================================
echo   RL-MEC Training Dashboard
echo ========================================
echo.

cd /d "%~dp0.."

echo [1/2] Checking dependencies...
where python >nul 2>nul || (
    echo [ERROR] Python not found. Please ensure Python is in PATH.
    echo        You can add the Python installation directory to PATH,
    echo        or use the venv in this project:
    echo        .\.venv\Scripts\python
    pause
    exit /b 1
)

python -c "import fastapi, uvicorn" 2>nul
if errorlevel 1 (
    echo [WARNING] fastapi/uvicorn not found, installing...
    python -m pip install fastapi uvicorn --quiet
)

echo.
echo [2/2] Starting dashboard server...
echo    Logs directory: logs
echo    URL: http://127.0.0.1:8088
echo    Press Ctrl+C to stop
echo.

python scripts\serve_dashboard.py --logs-dir logs --host 127.0.0.1 --port 8088

pause