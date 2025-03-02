"""Utility functions for enforcing stage constraints."""
import numpy as np
from typing import Optional
import logging

def enforce_stage_allocation(allocation: np.ndarray, max_stage_ratio: float = 0.8, 
                           min_stage_ratio: float = 0.05, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Enforce stage allocation constraints by clipping and redistributing delta-V.
    
    Args:
        allocation: Array of stage delta-V allocations
        max_stage_ratio: Maximum fraction of total delta-V allowed in any stage (default: 0.8)
        min_stage_ratio: Minimum fraction required for each stage (default: 0.05)
        logger: Optional logger for debug messages
        
    Returns:
        Rebalanced allocation array meeting constraints
    """
    new_alloc = allocation.copy()
    total_dv = np.sum(new_alloc)
    max_dv = max_stage_ratio * total_dv
    min_dv = min_stage_ratio * total_dv
    
    # First enforce maximum constraint
    while np.any(new_alloc > max_dv):
        for i in range(len(new_alloc)):
            if new_alloc[i] > max_dv:
                if logger:
                    logger.debug(f"Stage {i} exceeds max ratio: {new_alloc[i]/total_dv:.3f}")
                
                # Calculate excess and clip
                excess = new_alloc[i] - max_dv
                new_alloc[i] = max_dv
                
                # Redistribute excess proportionally to other stages
                other_stages = [j for j in range(len(new_alloc)) if j != i]
                other_sum = np.sum(new_alloc[other_stages])
                
                if other_sum > 1e-10:
                    # Proportional redistribution
                    redistribution = new_alloc[other_stages] / other_sum * excess
                else:
                    # Equal redistribution if others are near zero
                    redistribution = np.full(len(other_stages), excess / len(other_stages))
                
                new_alloc[other_stages] += redistribution
                if logger:
                    logger.debug(f"Redistributed {excess:.3f} delta-V from stage {i}")
    
    # Then enforce minimum constraint
    while np.any(new_alloc < min_dv):
        for i in range(len(new_alloc)):
            if new_alloc[i] < min_dv:
                if logger:
                    logger.debug(f"Stage {i} below min ratio: {new_alloc[i]/total_dv:.3f}")
                
                # Calculate needed boost
                needed = min_dv - new_alloc[i]
                new_alloc[i] = min_dv
                
                # Take proportionally from other stages
                other_stages = [j for j in range(len(new_alloc)) if j != i and new_alloc[j] > min_dv]
                if not other_stages:
                    # If no stages above min, take equally from all others
                    other_stages = [j for j in range(len(new_alloc)) if j != i]
                
                other_sum = np.sum(new_alloc[other_stages])
                reduction = new_alloc[other_stages] / other_sum * needed
                new_alloc[other_stages] -= reduction
                
                if logger:
                    logger.debug(f"Boosted stage {i} by {needed:.3f} delta-V")
    
    # Normalize to maintain total delta-V
    new_alloc = new_alloc * (total_dv / np.sum(new_alloc))
    
    return new_alloc
