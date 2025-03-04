"""Test suite for rocket stage optimization constraints."""
from src.optimization.objective import payload_fraction_objective, objective_with_penalty
import numpy as np

def run_test(name, result, expected):
    print(f"\n{name}")
    print("-" * 40)
    print(f"Result:   {result}")
    print(f"Expected: {expected}")

def test_stage_constraints():
    G0 = 9.81
    
    print("\nTesting rocket stage optimization constraints...")
    print("=" * 40)
    
    # Test 1: Nonphysical case
    dv = np.array([10000, 5000])
    ISP = np.array([300, 300])
    EPSILON = np.array([0.1, 0.1])
    result = payload_fraction_objective(dv, G0, ISP, EPSILON)
    run_test("Test 1: Nonphysical Payload", result, "inf")
    
    # Test 2: Valid case
    dv = np.array([5000, 3000])
    ISP = np.array([300, 350])
    EPSILON = np.array([0.1, 0.08])
    result = payload_fraction_objective(dv, G0, ISP, EPSILON)
    run_test("Test 2: Valid Payload", result, "negative finite")
    
    # Test 3: Major violation
    dv = np.array([6000, 4000])
    TOTAL_DELTA_V = 8000
    result = objective_with_penalty(dv, G0, ISP, EPSILON, TOTAL_DELTA_V)
    run_test("Test 3: Major DeltaV Violation", result, "inf")
    
    # Test 4: Minor violation
    dv = np.array([4100, 4000])
    result = objective_with_penalty(dv, G0, ISP, EPSILON, TOTAL_DELTA_V)
    run_test("Test 4: Minor DeltaV Violation", result, "finite but penalized")
    
    # Test 5: Stage imbalance
    dv = np.array([7500, 500])
    result = objective_with_penalty(dv, G0, ISP, EPSILON, TOTAL_DELTA_V)
    run_test("Test 5: Stage Ratio Imbalance", result, "heavily penalized")
    
    # Test 6: Optimal solution
    dv = np.array([4500, 3500])
    result = objective_with_penalty(dv, G0, ISP, EPSILON, TOTAL_DELTA_V)
    run_test("Test 6: Optimal Solution", result, "best payload fraction")
    
    # Test 7: Low ISP
    ISP = np.array([200, 200])
    result = objective_with_penalty(dv, G0, ISP, EPSILON, TOTAL_DELTA_V)
    run_test("Test 7: Low ISP Edge Case", result, "poor but valid fraction")
    
    # Test 8: High structural mass
    ISP = np.array([300, 350])
    EPSILON = np.array([0.2, 0.2])
    result = objective_with_penalty(dv, G0, ISP, EPSILON, TOTAL_DELTA_V)
    run_test("Test 8: High Structural Mass", result, "reduced payload fraction")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    test_stage_constraints()
