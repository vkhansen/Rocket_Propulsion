"""Simple test for objective function constraints."""
from src.optimization.objective import payload_fraction_objective, objective_with_penalty
import numpy as np

def main():
    # Test 1: Nonphysical payload fraction
    print("\nTest 1: Nonphysical payload fraction")
    dv = np.array([10000, 5000])  # Unrealistically high delta-v
    G0 = 9.81
    ISP = np.array([300, 300])
    EPSILON = np.array([0.1, 0.1])
    
    result = payload_fraction_objective(dv, G0, ISP, EPSILON)
    print(f"Result (should be inf): {result}")
    
    # Test 2: Valid payload fraction
    print("\nTest 2: Valid payload fraction")
    dv = np.array([5000, 3000])
    ISP = np.array([300, 350])
    EPSILON = np.array([0.1, 0.08])
    
    result = payload_fraction_objective(dv, G0, ISP, EPSILON)
    print(f"Result (should be negative finite): {result}")
    
    # Test 3: Constraint violation
    print("\nTest 3: Constraint violation")
    dv = np.array([6000, 4000])
    TOTAL_DELTA_V = 8000
    
    result = objective_with_penalty(dv, G0, ISP, EPSILON, TOTAL_DELTA_V)
    print(f"Result (should be inf): {result}")

if __name__ == "__main__":
    main()
