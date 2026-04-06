@echo off
:: =========================================================================
:: House Price Prediction - One-Click Launcher (Windows)
:: =========================================================================

echo Starting House Price Prediction System...

:: 1. Start the FastAPI server in a new minimized window
echo [OK] Starting Backend API...
start "House Price API (Backend)" /min venv\Scripts\python.exe src\api\main.py

echo.
echo [OK] Waiting for API to initialize (5s)...
timeout /t 5 /nobreak >nul

:: 2. Start the Streamlit dashboard
echo [OK] Starting Dashboard UI...
echo.
echo =========================================================================
echo DO NOT CLOSE THIS WINDOW WHILE USING THE DASHBOARD.
echo Close this window to exit the application.
echo =========================================================================
echo.

venv\Scripts\python.exe -m streamlit run src\dashboard\app.py

:: 3. Cleanup after exit
echo Exiting...
taskkill /F /FI "WINDOWTITLE eq House Price API (Backend)" /T >nul 2>&1
pause
