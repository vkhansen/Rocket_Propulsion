"""Configuration module."""
import os
import json
import logging
import sys
from logging.handlers import RotatingFileHandler

# Create output directory if it doesn't exist
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure logging
def setup_logging():
    """Set up logging configuration with multiple handlers and formats."""
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Console handler - for basic output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Debug file handler - for detailed debug information
    debug_handler = RotatingFileHandler(
        os.path.join(OUTPUT_DIR, "debug.log"),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        mode='w'
    )
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(detailed_formatter)
    
    # Optimization file handler - for optimization-specific info
    opt_handler = RotatingFileHandler(
        os.path.join(OUTPUT_DIR, "optimization.log"),
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3,
        mode='w'
    )
    opt_handler.setLevel(logging.INFO)
    opt_handler.setFormatter(detailed_formatter)
    
    # Error file handler - for warnings and errors
    error_handler = RotatingFileHandler(
        os.path.join(OUTPUT_DIR, "error.log"),
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3,
        mode='w'
    )
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(detailed_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # Remove any existing handlers
    root_logger.handlers = []
    
    # Add all handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(debug_handler)
    root_logger.addHandler(opt_handler)
    root_logger.addHandler(error_handler)

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

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

def load_config():
    """Load configuration from config.json."""
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise

# Try to load user configuration
try:
    user_config = load_config()
    # Update configuration with user settings
    for key in user_config:
        if key in CONFIG["optimization"]:
            CONFIG["optimization"][key].update(user_config[key])
except Exception as e:
    logger.warning(f"Error loading user configuration: {e}")

# Global variables that will be set by load_input_data
TOTAL_DELTA_V = None
