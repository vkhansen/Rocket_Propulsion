"""Physics calculations for rocket stage optimization."""
import numpy as np
from ..utils.config import logger

def calculate_mass_ratios(dv, ISP, EPSILON, G0):
    """Calculate mass ratios for each stage.
    
    Args:
        dv: Stage delta-v values
        ISP: Specific impulse values
        EPSILON: Structural fraction values
        G0: Gravitational constant
        
    Returns:
        numpy.ndarray: Mass ratios for each stage
    """
    try:
        # Convert inputs to numpy arrays
        dv = np.asarray(dv, dtype=float)
        ISP = np.asarray(ISP, dtype=float)
        EPSILON = np.asarray(EPSILON, dtype=float)
        
        # Calculate mass ratios using the rocket equation
        # r = exp(-ΔV/(g₀*ISP))
        mass_ratios = np.exp(-dv / (ISP * G0))
        
        return mass_ratios
        
    except Exception as e:
        logger.error(f"Error calculating mass ratios: {e}")
        return np.ones_like(dv)  # Return neutral mass ratios on error

def calculate_payload_fraction(mass_ratios, EPSILON):
    """Calculate payload fraction from mass ratios.
    
    Args:
        mass_ratios: Mass ratios for each stage
        EPSILON: Structural fraction values
        
    Returns:
        float: Payload fraction
    """
    try:
        # Convert inputs to numpy arrays
        mass_ratios = np.asarray(mass_ratios, dtype=float)
        EPSILON = np.asarray(EPSILON, dtype=float)
        
        # Initialize payload fraction as 1
        payload_fraction = 1.0
        
        # Calculate payload fraction stage by stage
        # For each stage i: Λᵢ = rᵢ/(1 + εᵢ)
        for i in range(len(mass_ratios)):
            stage_ratio = mass_ratios[i] / (1.0 + EPSILON[i])
            payload_fraction *= stage_ratio
        
        return float(payload_fraction)
        
    except Exception as e:
        logger.error(f"Error calculating payload fraction: {e}")
        return 0.0  # Return zero payload fraction on error

def calculate_stage_ratios(dv, G0, ISP, EPSILON):
    """Calculate stage ratios (Λ) for each stage.
    
    Args:
        dv: Stage delta-v values
        G0: Gravitational constant
        ISP: Specific impulse values
        EPSILON: Structural fraction values
        
    Returns:
        tuple: (stage_ratios, mass_ratios)
            - stage_ratios: List of stage ratios (Λ)
            - mass_ratios: List of mass ratios (r)
            
    For each stage i:
    - rᵢ = exp(-ΔVᵢ/(g₀*ISPᵢ))  # Mass ratio from rocket equation
    - Λᵢ = rᵢ/(1 + εᵢ)           # Stage ratio
    """
    try:
        # Calculate mass ratios
        mass_ratios = calculate_mass_ratios(dv, ISP, EPSILON, G0)
        
        # Calculate stage ratios (Λ)
        stage_ratios = []
        for i in range(len(mass_ratios)):
            lambda_i = mass_ratios[i] / (1.0 + EPSILON[i])
            stage_ratios.append(float(lambda_i))
            
        return np.array(stage_ratios), mass_ratios
        
    except Exception as e:
        logger.error(f"Error calculating stage ratios: {e}")
        return np.ones_like(dv), np.ones_like(dv)
