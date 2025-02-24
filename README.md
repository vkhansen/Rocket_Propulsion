# Rocket Propulsion Optimization

A Python-based tool for optimizing multi-stage rocket propulsion systems using various optimization algorithms including Particle Swarm Optimization (PSO), Differential Evolution (DE), and Genetic Algorithms (GA).

## Installation and Setup Guide

### Prerequisites

1. **Visual Studio Code (VSCode)**
   - Download from: [https://code.visualstudio.com/](https://code.visualstudio.com/)
   - Install the Python extension:
     1. Open VSCode
     2. Click the Extensions icon in the sidebar (or press Ctrl+Shift+X)
     3. Search for "Python"
     4. Install the Microsoft Python extension

2. **GitHub Desktop**
   - Download from: [https://desktop.github.com/](https://desktop.github.com/)
   - Install and sign in with your GitHub account

3. **Python 3.8 or higher**
   - Download from: [https://www.python.org/downloads/](https://www.python.org/downloads/)
   - **Important**: During installation, check "Add Python to PATH"

### Setting Up the Project

1. **Clone the Repository**
   - Open GitHub Desktop
   - Click "Clone a Repository"
   - URL: `https://github.com/vkhansen/rocket_propulsion.git`
   - Choose your local path
   - Click "Clone"

2. **Open in VSCode**
   - In GitHub Desktop, click "Open in Visual Studio Code"
   - Or open VSCode and select File > Open Folder > [repository location]

3. **Setup Python Environment**

   **For Windows:**
   ```bash
   # Open a terminal in VSCode (Ctrl+Shift+`)
   # Run the setup script
   setup_env_win.bat
   # Activate the environment
   venv\Scripts\activate
   ```

   **For Mac/Linux:**
   ```bash
   # Open a terminal in VSCode (Cmd+Shift+`)
   # Make the script executable
   chmod +x setup_env_mac.sh
   # Run the setup script
   ./setup_env_mac.sh
   # Activate the environment
   source venv/bin/activate
   ```

### Running the Code

1. **Run Tests**
   ```bash
   pytest Stage_Opt/test_payload_optimization.py
   ```

2. **Run Optimization**
   ```bash
   python Stage_Opt/main.py
   ```

3. **View Results**
   - Check the `Stage_Opt/output` directory for:
     - Optimization results (CSV)
     - LaTeX reports
     - Visualization plots
   - Check `Stage_Opt/logs` for detailed logging information

## Project Structure

```
rocket_propulsion/
├── Stage_Opt/
│   ├── src/
│   │   ├── optimization/  # Optimization algorithms
│   │   ├── reporting/     # LaTeX report generation
│   │   ├── utils/         # Configuration and utilities
│   │   └── visualization/ # Plotting functions
│   ├── output/            # Results and reports
│   ├── logs/             # Log files
│   └── test_payload_optimization.py
├── setup_env_win.bat     # Windows setup script
└── setup_env_mac.sh      # Mac setup script
```

## Troubleshooting

1. **Python not found**
   - Ensure Python is added to PATH during installation
   - Try running `python --version` or `python3 --version`

2. **pip not found**
   - Try running `python -m pip --version`
   - If missing, download [get-pip.py](https://bootstrap.pypa.io/get-pip.py) and run:
     ```bash
     python get-pip.py
     ```

3. **VSCode Python Interpreter**
   - Press Ctrl+Shift+P (Cmd+Shift+P on Mac)
   - Type "Python: Select Interpreter"
   - Choose the interpreter in the virtual environment (venv)

4. **Permission Issues (Mac/Linux)**
   - If you get permission errors:
     ```bash
     chmod +x setup_env_mac.sh
     ```

## Contributing

1. Create a new branch in GitHub Desktop
2. Make your changes
3. Commit with a descriptive message
4. Create a Pull Request

## Support

For issues or questions:
1. Check the logs in `Stage_Opt/logs`
2. Create an issue on GitHub
3. Contact the maintainers
