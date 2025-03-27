"""Configuration module."""
import os
import json
import logging
import sys
from datetime import datetime

# Constants
CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config.json')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'output')

def load_config():
    """Load configuration from config.json.
    
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger = setup_logging()
        logger.error(f"Failed to load config: {str(e)}")
        return {}

def setup_logging(solver_name=None):
    """Set up logging configuration with console output only for improved performance."""
    # Create unique logger for this solver instance
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if solver_name:
        log_name = f"{solver_name}_{timestamp}"
        logger = logging.getLogger(log_name)
    else:
        log_name = "optimization"
        logger = logging.getLogger("optimization")
    
    # Only add handlers if logger doesn't have any
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Console handler only
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger

# Initialize logging
logger = setup_logging()
logger.setLevel(logging.INFO)  # Changed from DEBUG to INFO to reduce verbosity

# Console handler - for basic output
# This is the only handler we'll use for performance reasons
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# Default configuration
CONFIG = {
    "optimization": {
        "tolerance": 1e-6,
        "max_iterations": 1000,
        "slsqp": {
            "maxiter": 1000,
            "ftol": 1e-6,
            "eps": 1e-8
        },
        "basin_hopping": {
            "niter": 100,
            "T": 1.0,
            "stepsize": 0.5,
            "maxiter": 1000,
            "ftol": 1e-6
        },
        "differential_evolution": {
            "maxiter": 1000,
            "popsize": 15,
            "mutation": (0.5, 1.0),
            "recombination": 0.7,
            "seed": 42,
            "strategy": "best1bin"
        },
        "ga": {
            "pop_size": 100,
            "n_generations": 100,
            "crossover_prob": 0.9,
            "mutation_prob": 0.1,
            "tournament_size": 3
        },
        "adaptive_ga": {
            "initial_pop_size": 100,
            "max_generations": 100,
            "initial_mutation_rate": 0.1,
            "initial_crossover_rate": 0.9,
            "min_mutation_rate": 0.01,
            "max_mutation_rate": 0.5,
            "min_crossover_rate": 0.5,
            "max_crossover_rate": 1.0,
            "tournament_size": 3,
            "elite_size": 2,
            "convergence_threshold": 1e-6,
            "stagnation_limit": 10
        },
        "pso": {
            "n_particles": 50,
            "n_iterations": 100,
            "c1": 2.0,  # Cognitive parameter
            "c2": 2.0,  # Social parameter
            "w": 0.7    # Inertia weight
        }
    }
}

# Load configuration
config = load_config()

# Try to load user configuration
try:
    # Update configuration with user settings
    for key in config:
        if key in CONFIG["optimization"]:
            CONFIG["optimization"][key].update(config[key])
except Exception as e:
    logger.warning(f"Error loading user configuration: {e}")

# Global variables that will be set by load_input_data
TOTAL_DELTA_V = None
