"""Configuration module."""
import os
import json
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join("output", "debug.log"), mode='w')
    ]
)
logger = logging.getLogger(__name__)

# Create output directory if it doesn't exist
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# Try to load user configuration
try:
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = json.load(f)
            # Update configuration with user settings
            for key in user_config:
                if key in CONFIG["optimization"]:
                    CONFIG["optimization"][key].update(user_config[key])
except Exception as e:
    logger.warning(f"Error loading user configuration: {e}")

# Global variables that will be set by load_input_data
TOTAL_DELTA_V = None
