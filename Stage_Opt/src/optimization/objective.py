"""Objective functions for optimization."""
import numpy as np
from ..utils.data import calculate_mass_ratios, calculate_payload_fraction
from ..utils.config import logger

def payload_fraction_objective(dv, G0, ISP, EPSILON):
    """Calculate the payload fraction objective using the corrected physics model."""
    try:
        # Pass G0 to calculate_mass_ratios so that the negative exponent is used
        mass_ratios = calculate_mass_ratios(dv, ISP, EPSILON, G0)
        payload_fraction = calculate_payload_fraction(mass_ratios)
        
        # Add a small penalty for solutions close to constraint violations
        penalty = 0.0
        for ratio in mass_ratios:
            if ratio <= 0.1:  # Penalize solutions close to physical limits
                penalty += 100.0 * (0.1 - ratio) ** 2
                
        # Negative for minimization
        return float(-payload_fraction + penalty)
    except Exception as e:
        logger.error(f"Error in payload fraction calculation: {e}")
        return 1e6  # Large but finite penalty

def objective_with_penalty(dv, G0, ISP, EPSILON, TOTAL_DELTA_V):
    """Calculate objective with penalty for constraint violation."""
    try:
        # Base objective
        base_obj = payload_fraction_objective(dv, G0, ISP, EPSILON)
        
        # Constraint violation penalty
        dv_sum = float(np.sum(dv))
        constraint_violation = abs(dv_sum - TOTAL_DELTA_V)
        penalty = 1e3 * constraint_violation  # Reduced penalty coefficient
        
        return float(base_obj + penalty)
    except Exception as e:
        logger.error(f"Error in objective calculation: {e}")
        return 1e6  # Large but finite penalty
