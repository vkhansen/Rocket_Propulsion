"""Physics calculations for rocket stage optimization."""
import numpy as np
from ..utils.config import logger

def calculate_stage_ratios(dv, G0, ISP, EPSILON):
    """Calculate stage and mass ratios for given delta-v values.
    
    Args:
        dv (np.ndarray): Delta-v values for each stage
        G0 (float): Gravitational acceleration
        ISP (np.ndarray): Specific impulse for each stage
        EPSILON (np.ndarray): Structural coefficients for each stage
        
    Returns:
        tuple: (stage_ratios, mass_ratios)
    """
    try:
        # Convert inputs to numpy arrays
        dv = np.asarray(dv, dtype=float)
        ISP = np.asarray(ISP, dtype=float)
        EPSILON = np.asarray(EPSILON, dtype=float)
        
        # Calculate stage ratios (λ)
        stage_ratios = np.exp(dv / (G0 * ISP))
        
        # Calculate mass ratios (μ)
        mass_ratios = np.zeros_like(stage_ratios)
        for i in range(len(stage_ratios)):
            mass_ratios[i] = stage_ratios[i] / (1.0 - EPSILON[i] * (1.0 - stage_ratios[i]))
        
        return stage_ratios, mass_ratios
        
    except Exception as e:
        logger.error(f"Error calculating stage ratios: {str(e)}")
        return np.ones_like(dv), np.ones_like(dv)

def calculate_mass_ratios(stage_ratios, EPSILON):
    """Calculate mass ratios from stage ratios.
    
    Args:
        stage_ratios (np.ndarray): Stage ratios (λ) for each stage
        EPSILON (np.ndarray): Structural coefficients for each stage
        
    Returns:
        np.ndarray: Mass ratios (μ) for each stage
    """
    try:
        # Convert inputs to numpy arrays
        stage_ratios = np.asarray(stage_ratios, dtype=float)
        EPSILON = np.asarray(EPSILON, dtype=float)
        
        # Calculate mass ratios
        mass_ratios = np.zeros_like(stage_ratios)
        for i in range(len(stage_ratios)):
            mass_ratios[i] = stage_ratios[i] / (1.0 - EPSILON[i] * (1.0 - stage_ratios[i]))
            
        return mass_ratios
        
    except Exception as e:
        logger.error(f"Error calculating mass ratios: {str(e)}")
        return np.ones_like(stage_ratios)

def calculate_payload_fraction(mass_ratios, EPSILON):
    """Calculate payload fraction from mass ratios.
    
    Args:
        mass_ratios (np.ndarray): Mass ratios (μ) for each stage
        EPSILON (np.ndarray): Structural coefficients for each stage
        
    Returns:
        float: Payload fraction
    """
    try:
        # Convert inputs to numpy arrays
        mass_ratios = np.asarray(mass_ratios, dtype=float)
        
        # Calculate payload fraction
        payload_fraction = 1.0
        for ratio in mass_ratios:
            payload_fraction /= ratio
            
        return float(payload_fraction)
        
    except Exception as e:
        logger.error(f"Error calculating payload fraction: {str(e)}")
        return 0.0
