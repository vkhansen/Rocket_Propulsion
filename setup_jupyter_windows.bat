@echo off
:: Check for Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python 3.8 or later from https://www.python.org/.
    pause
    exit /b
)

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install Jupyter Lab
echo Installing Jupyter Lab...
pip install jupyterlab

:: Install dependencies
echo Installing required dependencies...
pip install numpy pandas matplotlib scipy

:: Start Jupyter Lab
echo Starting Jupyter Lab...
jupyter lab

pause