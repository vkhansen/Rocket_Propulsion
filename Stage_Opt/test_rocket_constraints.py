"""Test suite for rocket stage optimization constraints."""
from src.optimization.objective import payload_fraction_objective, objective_with_penalty
import numpy as np

def print_test_header(name, description=None):
    print(f"\n{'='*80}")
    print(f"TEST: {name}")
    if description:
        print(f"Description: {description}")
    print('-'*80)

def test_stage_constraints():
    G0 = 9.81  # m/s²
    
    # Test 1: Nonphysical Payload Fraction
    print_test_header("Nonphysical Payload Fraction", 
                     "Testing rejection of solutions with impossible mass ratios")
    dv = np.array([10000, 5000])  # Unrealistically high delta-v
    ISP = np.array([300, 300])
    EPSILON = np.array([0.1, 0.1])
    result = payload_fraction_objective(dv, G0, ISP, EPSILON)
    print(f"Inputs: dv={dv}, ISP={ISP}, ε={EPSILON}")
    print(f"Result: {result} (Expected: inf)\n")

    # Test 2: Valid Payload Fraction
    print_test_header("Valid Payload Fraction",
                     "Testing acceptance of physically realistic solution")
    dv = np.array([5000, 3000])
    ISP = np.array([300, 350])
    EPSILON = np.array([0.1, 0.08])
    result = payload_fraction_objective(dv, G0, ISP, EPSILON)
    print(f"Inputs: dv={dv}, ISP={ISP}, ε={EPSILON}")
    print(f"Result: {result} (Expected: negative finite)\n")

    # Test 3: Major DeltaV Violation
    print_test_header("Major DeltaV Violation",
                     "Testing rejection of major constraint violation")
    dv = np.array([6000, 4000])  # 25% over limit
    TOTAL_DELTA_V = 8000
    result = objective_with_penalty(dv, G0, ISP, EPSILON, TOTAL_DELTA_V)
    print(f"Inputs: dv={dv}, Target ΔV={TOTAL_DELTA_V}")
    print(f"Violation: {(np.sum(dv) - TOTAL_DELTA_V)/TOTAL_DELTA_V*100:.1f}% over limit")
    print(f"Result: {result} (Expected: inf)\n")

    # Test 4: Minor DeltaV Violation
    print_test_header("Minor DeltaV Violation",
                     "Testing penalty for minor constraint violation")
    dv = np.array([4100, 4000])  # 1.25% over limit
    result = objective_with_penalty(dv, G0, ISP, EPSILON, TOTAL_DELTA_V)
    print(f"Inputs: dv={dv}, Target ΔV={TOTAL_DELTA_V}")
    print(f"Violation: {(np.sum(dv) - TOTAL_DELTA_V)/TOTAL_DELTA_V*100:.1f}% over limit")
    print(f"Result: {result} (Expected: finite but penalized)\n")

    # Test 5: Stage Ratio Imbalance
    print_test_header("Stage Ratio Imbalance",
                     "Testing penalty for poorly balanced stages")
    dv = np.array([7500, 500])  # Very unbalanced (93.75%/6.25%)
    result = objective_with_penalty(dv, G0, ISP, EPSILON, TOTAL_DELTA_V)
    print(f"Inputs: dv={dv}, Target ΔV={TOTAL_DELTA_V}")
    print(f"Stage split: {dv[0]/np.sum(dv)*100:.1f}% / {dv[1]/np.sum(dv)*100:.1f}%")
    print(f"Result: {result} (Expected: heavily penalized)\n")

    # Test 6: Optimal Solution
    print_test_header("Optimal Solution",
                     "Testing acceptance of well-balanced solution")
    dv = np.array([4500, 3500])  # Well balanced (56.25%/43.75%)
    result = objective_with_penalty(dv, G0, ISP, EPSILON, TOTAL_DELTA_V)
    print(f"Inputs: dv={dv}, Target ΔV={TOTAL_DELTA_V}")
    print(f"Stage split: {dv[0]/np.sum(dv)*100:.1f}% / {dv[1]/np.sum(dv)*100:.1f}%")
    print(f"Result: {result} (Expected: optimal finite value)\n")

    # Test 7: Low ISP Edge Case
    print_test_header("Low ISP Edge Case",
                     "Testing handling of poor engine efficiency")
    ISP_low = np.array([200, 200])
    result = objective_with_penalty(dv, G0, ISP_low, EPSILON, TOTAL_DELTA_V)
    print(f"Inputs: dv={dv}, ISP={ISP_low}, Target ΔV={TOTAL_DELTA_V}")
    print(f"Result: {result} (Expected: poor but valid payload fraction)\n")

    # Test 8: High Structural Mass
    print_test_header("High Structural Mass",
                     "Testing impact of high structural coefficients")
    EPSILON_high = np.array([0.2, 0.2])
    result = objective_with_penalty(dv, G0, ISP, EPSILON_high, TOTAL_DELTA_V)
    print(f"Inputs: dv={dv}, ε={EPSILON_high}, Target ΔV={TOTAL_DELTA_V}")
    print(f"Result: {result} (Expected: reduced payload fraction)\n")

if __name__ == "__main__":
    print("\nStarting rocket stage optimization constraint tests...")
    test_stage_constraints()
    print("\nAll tests completed!")
