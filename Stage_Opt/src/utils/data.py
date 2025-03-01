"""Data loading and processing utilities."""
import json
from .config import logger

def load_input_data(filename):
    """Load input data from JSON file.
    
    Args:
        filename: Path to JSON input file
        
    Returns:
        tuple: (parameters, stages)
            - parameters: Dictionary of global parameters
            - stages: List of stage configurations, sorted by stage number
    """
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            
        # Extract global parameters
        parameters = data['parameters']
        
        # Sort stages by stage number
        stages = sorted(data['stages'], key=lambda x: x['stage'])
        
        return parameters, stages
        
    except Exception as e:
        logger.error(f"Error loading input data: {e}")
        raise
