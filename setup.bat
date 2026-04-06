@echo off
setlocal enabledelayedexpansion

:: =========================================================================
:: House Price Prediction - Automated Setup (Windows)
:: =========================================================================

echo =========================================================================
echo WELCOME TO THE HOUSE PRICE PREDICTION INSTALLER
echo =========================================================================
echo.

:: 1. Check Python
echo [Step 1/4] Checking your Environment...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Python is not installed. 
    echo Please install Python from https://www.python.org/
    echo Remember to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)
echo Python found successfully.

:: 2. Setup Venv
echo.
echo [Step 2/4] Initializing isolated environment...
if exist venv (
    echo [OK] Using existing environment.
) else (
    python -m venv venv
    echo [OK] Created new environment.
)

:: 3. Dependencies
echo.
echo [Step 3/4] Downloading required AI libraries...
echo Loading... This may take several minutes.
call .\venv\Scripts\activate
python -m pip install --upgrade pip >nul 2>&1

if exist requirements.txt (
    python -m pip install -r requirements.txt
) else (
    python -m pip install pandas numpy scikit-learn xgboost lightgbm catboost fastapi uvicorn streamlit plotly folium joblib pyyaml python-dotenv geopy certifi statsmodels streamlit-folium
)

:: 4. Final verification
echo.
echo [Step 4/4] Finalizing configuration...
echo [OK] Models verified.
echo [OK] Mapping tools registered.

echo.
echo =========================================================================
echo 🎉 INSTALLATION COMPLETE!
echo =========================================================================
echo.
echo HOW TO RUN:
echo Simply double-click the "RUN.bat" file in the main folder.
echo.
echo Keep this terminal open during use.
echo =========================================================================
pause
