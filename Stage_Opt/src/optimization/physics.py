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
        tuple: (stage_ratios, mass_ratios) where:
            - stage_ratios (λ) = mf/m0 (final mass / initial mass for each stage)
            - mass_ratios (μ) = stage mass ratio accounting for structural mass
    """
    try:
        # Convert inputs to numpy arrays
        dv = np.asarray(dv, dtype=float)
        ISP = np.asarray(ISP, dtype=float)
        EPSILON = np.asarray(EPSILON, dtype=float)
        
        # Calculate stage ratios (λ = mf/m0)
        stage_ratios = np.exp(-dv / (G0 * ISP))  # Added negative sign to get mf/m0
        
        # Calculate mass ratios (μ)
        mass_ratios = np.zeros_like(stage_ratios)
        for i in range(len(stage_ratios)):
            # Since λ is now mf/m0, we need to adjust the mass ratio formula
            mass_ratios[i] = 1.0 / (stage_ratios[i] * (1.0 - EPSILON[i]) + EPSILON[i])
        
        return stage_ratios, mass_ratios
        
    except Exception as e:
        logger.error(f"Error calculating stage ratios: {str(e)}")
        return np.ones_like(dv), np.ones_like(dv)

def calculate_mass_ratios(stage_ratios, EPSILON):
    """Calculate mass ratios from stage ratios.
    
    Args:
        stage_ratios (np.ndarray): Stage ratios (λ = mf/m0) for each stage
        EPSILON (np.ndarray): Structural coefficients for each stage
        
    Returns:
        np.ndarray: Mass ratios (μ) for each stage
    """
    try:
        # Convert inputs to numpy arrays
        stage_ratios = np.asarray(stage_ratios, dtype=float)
        EPSILON = np.asarray(EPSILON, dtype=float)
        
        # Calculate mass ratios using corrected formula for mf/m0
        mass_ratios = np.zeros_like(stage_ratios)
        for i in range(len(stage_ratios)):
            mass_ratios[i] = 1.0 / (stage_ratios[i] * (1.0 - EPSILON[i]) + EPSILON[i])
            
        return mass_ratios
        
    except Exception as e:
        logger.error(f"Error calculating mass ratios: {str(e)}")
        return np.ones_like(stage_ratios)

def calculate_payload_fraction(mass_ratios):
    """Calculate payload fraction from mass ratios.
    
    Args:
        mass_ratios (np.ndarray): Mass ratios (μ) for each stage
        
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
