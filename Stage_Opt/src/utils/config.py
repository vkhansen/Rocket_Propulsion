"""Configuration and logging setup."""
import os
import json
import logging

# Get the directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define paths
CONFIG_FILE = os.path.join(SCRIPT_DIR, 'config.json')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
LOG_FILE = os.path.join(OUTPUT_DIR, 'optimization.log')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load configuration
try:
    with open(CONFIG_FILE, 'r') as f:
        CONFIG = json.load(f)
except Exception as e:
    print(f"Error loading config file: {e}")
    CONFIG = {
        "optimization": {
            "tolerance": 1e-6,
            "max_iterations": 200,
            "penalty_coefficient": 1000.0,
            "basin_hopping": {
                "n_iterations": 100,
                "temperature": 1.0,
                "step_size": 0.5
            },
            "differential_evolution": {
                "strategy": "best1bin",
                "max_iterations": 1000,
                "population_size": 15,
                "tol": 1e-6,
                "mutation": [0.5, 1.0],
                "recombination": 0.7
            },
            "genetic_algorithm": {
                "population_size": 100,
                "max_generations": 200,
                "xtol": 1e-6,
                "ftol": 1e-6,
                "period": 20
            }
        }
    }

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode='w')
    ]
)

# Get logger
logger = logging.getLogger(__name__)
