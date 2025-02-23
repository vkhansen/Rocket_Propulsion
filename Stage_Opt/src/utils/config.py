"""Configuration and logging setup."""
import os
import json
import logging

def load_config():
    """Load configuration from config.json."""
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config.json')
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load config.json: {e}")
        raise

def setup_logging(config):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, config["logging"]["level"]),
        format=config["logging"]["format"],
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'output', 'optimization.log'), mode='w')
        ]
    )
    return logging.getLogger(__name__)

# Initialize globals from config
CONFIG = load_config()
logger = setup_logging(CONFIG)

# Ensure output directory exists
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global variables that will be set by load_input_data
TOTAL_DELTA_V = None
