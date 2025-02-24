#!/bin/bash

echo "Setting up Python environment for Rocket Propulsion Optimization..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed! Please install Python 3.8 or higher."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "pip3 is not installed! Please install pip3."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install required packages
echo "Installing required packages..."
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

# Create necessary directories
mkdir -p Stage_Opt/output
mkdir -p Stage_Opt/logs

echo "Environment setup complete!"
echo "To activate the environment, run: source venv/bin/activate"
echo "To run tests: pytest Stage_Opt/test_payload_optimization.py"
echo "To run optimization: python Stage_Opt/main.py"
