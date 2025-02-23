"""Data loading and processing utilities."""
import json
import numpy as np
from .config import logger

def load_input_data(filename):
    """Load input data from JSON file."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            
        # Extract global parameters
        parameters = data['parameters']
        total_delta_v = float(parameters['TOTAL_DELTA_V'])
        g0 = float(parameters.get('G0', 9.81))
        
        # Sort stages by stage number
        stages = sorted(data['stages'], key=lambda x: x['stage'])
        
        # Add global parameters to each stage if not present
        for stage in stages:
            if 'G0' not in stage:
                stage['G0'] = g0
        
        return parameters, stages
        
    except Exception as e:
        logger.error(f"Error loading input data: {e}")
        raise

def calculate_mass_ratios(dv, ISP, EPSILON, G0=9.81):
    """Calculate stage ratios for each stage using the correct negative exponent."""
    try:
        dv = np.asarray(dv).flatten()
        mass_ratios = []
        for i, dvi in enumerate(dv):
            # Use a negative exponent as in Code A
            ratio = np.exp(-dvi / (G0 * ISP[i])) - EPSILON[i]
            mass_ratios.append(float(ratio))
        return np.array(mass_ratios)
    except Exception as e:
        logger.error(f"Error calculating mass ratios: {e}")
        return np.array([float('inf')] * len(dv))

def calculate_payload_fraction(mass_ratios):
    """Calculate payload fraction as the product of stage ratios."""
    try:
        if any(r <= 0 for r in mass_ratios):
            return 0.0
        return float(np.prod(mass_ratios))
    except Exception as e:
        logger.error(f"Error calculating payload fraction: {e}")
        return 0.0
