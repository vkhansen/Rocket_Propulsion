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
        
        # Sort stages by stage number
        stages = sorted(data['stages'], key=lambda x: x['stage'])
        
        return parameters, stages
        
    except Exception as e:
        logger.error(f"Error loading input data: {e}")
        raise

def calculate_mass_ratios(dv, ISP, EPSILON, G0=9.81):
    """Calculate stage ratios (Λ) for each stage.
    
    Stage ratio Λᵢ = mᵢ₊₁/mᵢ is the ratio of the mass of the upper stage (i+1)
    to the mass of the current stage (i) before burnout.
    """
    try:
        dv = np.asarray(dv).flatten()
        stage_ratios = []
        
        # Calculate mass ratios in reverse (from top stage down)
        # since we need the mass of the upper stage to calculate the ratio
        for i in range(len(dv)-1, -1, -1):
            # Calculate mass ratio for current stage using rocket equation
            mass_ratio = np.exp(-dv[i] / (G0 * ISP[i]))
            
            if i > 0:  # For all stages except the bottom stage
                # Stage ratio Λᵢ = mᵢ₊₁/mᵢ
                # Upper stage mass includes structural mass
                upper_stage_mass = mass_ratio * (1 + EPSILON[i])
                stage_ratio = 1.0 / (upper_stage_mass + EPSILON[i-1])
                stage_ratios.insert(0, float(stage_ratio))
            else:  # Bottom stage
                stage_ratio = 1.0 / (mass_ratio + EPSILON[i])
                stage_ratios.insert(0, float(stage_ratio))
                
        return np.array(stage_ratios)
    except Exception as e:
        logger.error(f"Error calculating stage ratios: {e}")
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
