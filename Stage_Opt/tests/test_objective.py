"""Test objective function constraints and penalties."""
import numpy as np
import pytest
from Stage_Opt.src.optimization.objective import (
    payload_fraction_objective,
    objective_with_penalty,
    calculate_stage_ratios
)

def test_nonphysical_payload_fraction():
    """Test rejection of nonphysical payload fractions."""
    # Setup test case that would result in negative payload fraction
    dv = np.array([10000, 5000])  # Unrealistically high delta-v
    G0 = 9.81
    ISP = np.array([300, 300])
    EPSILON = np.array([0.1, 0.1])
    
    result = payload_fraction_objective(dv, G0, ISP, EPSILON)
    assert result == float('inf'), "Should reject nonphysical payload fraction with inf"

def test_valid_payload_fraction():
    """Test acceptance of valid payload fraction."""
    # Setup realistic test case
    dv = np.array([5000, 3000])
    G0 = 9.81
    ISP = np.array([300, 350])
    EPSILON = np.array([0.1, 0.08])
    
    result = payload_fraction_objective(dv, G0, ISP, EPSILON)
    assert result < 0, "Valid payload fraction should return negative value for minimization"
    assert result != float('-inf'), "Valid solution should not return -inf"
    assert result != float('inf'), "Valid solution should not return inf"

def test_constraint_penalties():
    """Test penalty application for constraint violations."""
    dv = np.array([6000, 4000])  # Total 10000, exceeding typical requirements
    G0 = 9.81
    ISP = np.array([300, 350])
    EPSILON = np.array([0.1, 0.08])
    TOTAL_DELTA_V = 8000  # Required total
    
    result = objective_with_penalty(dv, G0, ISP, EPSILON, TOTAL_DELTA_V)
    assert result == float('inf'), "Should heavily penalize significant dV violation"

def test_stage_ratio_constraints():
    """Test enforcement of stage ratio physical constraints."""
    # Setup case with invalid stage ratios
    dv = np.array([1000, 9000])  # Very unbalanced stages
    G0 = 9.81
    ISP = np.array([300, 350])
    EPSILON = np.array([0.1, 0.08])
    TOTAL_DELTA_V = 10000
    
    result = objective_with_penalty(dv, G0, ISP, EPSILON, TOTAL_DELTA_V)
    assert result > 1000, "Should apply strong penalty for unbalanced stages"
