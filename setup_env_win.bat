@echo off
echo Setting up Python environment for Rocket Propulsion Optimization...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed! Please install Python 3.8 or higher.
    exit /b 1
)

REM Check if pip is installed
pip --version >nul 2>&1
if errorlevel 1 (
    echo pip is not installed! Please install pip.
    exit /b 1
)

REM Check if pdflatex is installed
pdflatex --version >nul 2>&1
if errorlevel 1 (
    echo WARNING: pdflatex is not installed!
    echo To generate PDF reports, please install a LaTeX distribution:
    echo 1. Download MiKTeX from https://miktex.org/download
    echo 2. Run the installer
    echo 3. Choose "Install MiKTeX only for me"
    echo 4. Select "Always install missing packages on-the-fly"
    echo.
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate

REM Upgrade pip
python -m pip install --upgrade pip

REM Install required packages from requirements.txt
echo Installing required packages...
pip install -r requirements.txt

REM Create necessary directories
if not exist "Stage_Opt\output" mkdir Stage_Opt\output
if not exist "Stage_Opt\logs" mkdir Stage_Opt\logs

echo.
echo Environment setup complete!
echo To activate the environment, run: venv\Scripts\activate
echo To run tests: pytest Stage_Opt\test_payload_optimization.py
if errorlevel 1 (
    echo NOTE: PDF report generation will be skipped - LaTeX not found
)

pause