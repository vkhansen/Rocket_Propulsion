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

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate

REM Upgrade pip
python -m pip install --upgrade pip

REM Install required packages
echo Installing required packages...
pip install numpy>=1.20.0
pip install scipy>=1.7.0
pip install matplotlib>=3.4.0
pip install pymoo>=0.6.0
pip install pandas>=1.3.0
pip install pytest>=6.2.0
pip install deap>=1.3.1
pip install notebook>=6.4.0
pip install ipywidgets>=7.6.0
pip install tqdm>=4.62.0

REM Create necessary directories
if not exist "Stage_Opt\output" mkdir Stage_Opt\output
if not exist "Stage_Opt\logs" mkdir Stage_Opt\logs

echo Environment setup complete!
echo To activate the environment, run: venv\Scripts\activate
echo To run tests: pytest Stage_Opt\test_payload_optimization.py
echo To run optimization: python Stage_Opt\main.py

pause