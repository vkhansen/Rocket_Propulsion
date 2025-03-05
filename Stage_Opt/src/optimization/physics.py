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
        dv = np.asarray(dv, dtype=float)
        ISP = np.asarray(ISP, dtype=float)
        EPSILON = np.asarray(EPSILON, dtype=float)
        
        # Compute stage ratios (λ = mf/m0)
        stage_ratios = np.exp(-dv / (G0 * ISP))

        # Compute mass ratios (μ) using vectorized NumPy operations
        mass_ratios = (stage_ratios - EPSILON) / (1.0 - EPSILON)
        
        return stage_ratios, mass_ratios
        
    except Exception as e:
        logger.error(f"Error calculating stage ratios: {str(e)}")
        return np.full_like(dv, np.nan), np.full_like(dv, np.nan)  # Use NaN instead of 1s

def calculate_payload_fraction(mass_ratios):
    """Calculate payload fraction from mass ratios.
    
    Args:
        mass_ratios (np.ndarray): Mass ratios (μ) for each stage
        
    Returns:
        float: Payload fraction
    """
    try:
        mass_ratios = np.asarray(mass_ratios, dtype=float)
        return np.prod(mass_ratios)  # Vectorized computation
        
    except Exception as e:
        logger.error(f"Error calculating payload fraction: {str(e)}")
        return np.nan  # Use NaN instead of 0.0
