"""Test objective function constraints and penalties."""
from src.optimization.objective import payload_fraction_objective, objective_with_penalty
import numpy as np

def format_test_output(test_name, inputs, result, expected):
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"{'-'*70}")
    print("INPUTS:")
    for key, value in inputs.items():
        print(f"  {key}: {value}")
    print(f"\nRESULT: {result}")
    print(f"EXPECTED: {expected}")
    print(f"{'='*70}\n")

def main():
    G0 = 9.81
    
    print("\nStarting constraint enforcement tests...\n")
    
    # Test 1: Nonphysical payload fraction (extreme case)
    dv = np.array([10000, 5000])  # Unrealistically high delta-v
    ISP = np.array([300, 300])
    EPSILON = np.array([0.1, 0.1])
    
    result = payload_fraction_objective(dv, G0, ISP, EPSILON)
    format_test_output(
        "Nonphysical Payload Fraction (Extreme)",
        {"dv": dv, "ISP": ISP, "EPSILON": EPSILON, "G0": G0},
        result,
        "inf (should reject nonphysical payload fraction)"
    )
    
    # Test 2: Valid payload fraction (realistic case)
    dv = np.array([5000, 3000])
    ISP = np.array([300, 350])
    EPSILON = np.array([0.1, 0.08])
    
    result = payload_fraction_objective(dv, G0, ISP, EPSILON)
    format_test_output(
        "Valid Payload Fraction (Realistic)",
        {"dv": dv, "ISP": ISP, "EPSILON": EPSILON, "G0": G0},
        result,
        "negative finite value (should be physically realistic)"
    )
    
    # Test 3: Major constraint violation (25% over)
    dv = np.array([6000, 4000])
    TOTAL_DELTA_V = 8000
    
    result = objective_with_penalty(dv, G0, ISP, EPSILON, TOTAL_DELTA_V)
    format_test_output(
        "Major DeltaV Constraint Violation",
        {"dv": dv, "total_dv": TOTAL_DELTA_V, "violation": "25% over limit"},
        result,
        "inf (should reject major violation)"
    )
    
    # Test 4: Minor constraint violation (1.25% over)
    dv = np.array([4100, 4000])
    TOTAL_DELTA_V = 8000
    
    result = objective_with_penalty(dv, G0, ISP, EPSILON, TOTAL_DELTA_V)
    format_test_output(
        "Minor DeltaV Constraint Violation",
        {"dv": dv, "total_dv": TOTAL_DELTA_V, "violation": "1.25% over limit"},
        result,
        "finite but penalized value"
    )
    
    # Test 5: Stage ratio imbalance (93.75%/6.25% split)
    dv = np.array([7500, 500])
    TOTAL_DELTA_V = 8000
    
    result = objective_with_penalty(dv, G0, ISP, EPSILON, TOTAL_DELTA_V)
    format_test_output(
        "Extreme Stage Ratio Imbalance",
        {"dv": dv, "total_dv": TOTAL_DELTA_V, "stage_split": "93.75%/6.25%"},
        result,
        "heavily penalized value or inf"
    )
    
    # Test 6: Perfect solution (56.25%/43.75% split)
    dv = np.array([4500, 3500])
    TOTAL_DELTA_V = 8000
    
    result = objective_with_penalty(dv, G0, ISP, EPSILON, TOTAL_DELTA_V)
    format_test_output(
        "Optimal Solution",
        {"dv": dv, "total_dv": TOTAL_DELTA_V, "stage_split": "56.25%/43.75%"},
        result,
        "finite optimal value (should be best payload fraction)"
    )
    
    # Test 7: Edge case with very low ISP
    dv = np.array([4000, 4000])
    ISP = np.array([200, 200])  # Very low ISP
    TOTAL_DELTA_V = 8000
    
    result = objective_with_penalty(dv, G0, ISP, EPSILON, TOTAL_DELTA_V)
    format_test_output(
        "Edge Case - Low ISP",
        {"dv": dv, "ISP": ISP, "total_dv": TOTAL_DELTA_V},
        result,
        "should handle low ISP case appropriately"
    )
    
    # Test 8: Edge case with high structural mass
    EPSILON = np.array([0.2, 0.2])  # High structural coefficients
    ISP = np.array([300, 350])  # Back to normal ISP
    
    result = objective_with_penalty(dv, G0, ISP, EPSILON, TOTAL_DELTA_V)
    format_test_output(
        "Edge Case - High Structural Mass",
        {"dv": dv, "EPSILON": EPSILON, "total_dv": TOTAL_DELTA_V},
        result,
        "should handle high structural mass appropriately"
    )
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()
