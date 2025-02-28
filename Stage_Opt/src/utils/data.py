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
    
    For each stage i, the stage ratio Λᵢ is calculated as:
    Λᵢ = rᵢ/(1 + εᵢ)
    
    where:
    - rᵢ is the mass ratio from the rocket equation: exp(-ΔVᵢ/(g₀*ISPᵢ))
    - εᵢ is the structural coefficient for stage i
    """
    try:
        dv = np.asarray(dv).flatten()
        stage_ratios = []
        
        # Calculate stage ratios for each stage
        for i in range(len(dv)):
            # Calculate mass ratio using rocket equation
            mass_ratio = np.exp(-dv[i] / (G0 * ISP[i]))
            
            # Calculate stage ratio consistently for all stages
            stage_ratio = mass_ratio / (1.0 + EPSILON[i])
            stage_ratios.append(float(stage_ratio))
                
        return np.array(stage_ratios)
    except Exception as e:
        logger.error(f"Error calculating stage ratios: {e}")
        return np.array([float('inf')] * len(dv))

def calculate_payload_fraction(stage_ratios):
    """Calculate payload fraction as the product of stage ratios.
    
    The payload fraction is the product of all stage ratios:
    PF = ∏ᵢ Λᵢ = ∏ᵢ (rᵢ/(1 + εᵢ))
    """
    try:
        if any(r <= 0 for r in stage_ratios):
            return 0.0
        return float(np.prod(stage_ratios))
    except Exception as e:
        logger.error(f"Error calculating payload fraction: {e}")
        return 0.0
