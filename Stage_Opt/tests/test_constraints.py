"""Test objective function constraints and penalties."""
from src.optimization.objective import payload_fraction_objective, objective_with_penalty
import numpy as np

def run_test(name, dv, G0, ISP, EPSILON, TOTAL_DELTA_V=None, description=None):
    """Run a single test case and format its output."""
    print(f"\n{'='*80}")
    print(f"TEST {name}")
    print(f"{'-'*80}")
    
    if description:
        print(f"Description: {description}")
    
    print("\nInputs:")
    print(f"  Delta-V:     {dv} m/s")
    print(f"  ISP:         {ISP} s")
    print(f"  Epsilon:     {EPSILON}")
    print(f"  G0:          {G0} m/s²")
    if TOTAL_DELTA_V is not None:
        print(f"  Target ΔV:   {TOTAL_DELTA_V} m/s")
        print(f"  Actual ΔV:   {np.sum(dv)} m/s")
        print(f"  Difference:  {(np.sum(dv) - TOTAL_DELTA_V)/TOTAL_DELTA_V*100:.2f}%")
    
    if TOTAL_DELTA_V is None:
        result = payload_fraction_objective(dv, G0, ISP, EPSILON)
    else:
        result = objective_with_penalty(dv, G0, ISP, EPSILON, TOTAL_DELTA_V)
    
    print(f"\nResult: {result}")
    print(f"{'='*80}")
    return result

def main():
    G0 = 9.81
    print("\nRunning constraint enforcement tests...")
    
    # Test Set 1: Basic Physical Constraints
    print("\nTest Set 1: Basic Physical Constraints")
    print("-" * 40)
    
    # 1.1: Extreme nonphysical case
    run_test("1.1 - Extreme Nonphysical",
             dv=np.array([10000, 5000]),
             G0=G0,
             ISP=np.array([300, 300]),
             EPSILON=np.array([0.1, 0.1]),
             description="Should reject due to nonphysical payload fraction")
    
    # 1.2: Realistic case
    run_test("1.2 - Realistic Case",
             dv=np.array([5000, 3000]),
             G0=G0,
             ISP=np.array([300, 350]),
             EPSILON=np.array([0.1, 0.08]),
             description="Should accept with reasonable payload fraction")
    
    # Test Set 2: Delta-V Constraints
    print("\nTest Set 2: Delta-V Constraints")
    print("-" * 40)
    
    # 2.1: Major violation
    run_test("2.1 - Major DeltaV Violation",
             dv=np.array([6000, 4000]),
             G0=G0,
             ISP=np.array([300, 350]),
             EPSILON=np.array([0.1, 0.08]),
             TOTAL_DELTA_V=8000,
             description="Should reject (+25% over limit)")
    
    # 2.2: Minor violation
    run_test("2.2 - Minor DeltaV Violation",
             dv=np.array([4100, 4000]),
             G0=G0,
             ISP=np.array([300, 350]),
             EPSILON=np.array([0.1, 0.08]),
             TOTAL_DELTA_V=8000,
             description="Should penalize but not reject (+1.25% over)")
    
    # Test Set 3: Stage Balance
    print("\nTest Set 3: Stage Balance")
    print("-" * 40)
    
    # 3.1: Extreme imbalance
    run_test("3.1 - Extreme Imbalance",
             dv=np.array([7500, 500]),
             G0=G0,
             ISP=np.array([300, 350]),
             EPSILON=np.array([0.1, 0.08]),
             TOTAL_DELTA_V=8000,
             description="Should reject (93.75%/6.25% split)")
    
    # 3.2: Optimal balance
    run_test("3.2 - Optimal Balance",
             dv=np.array([4500, 3500]),
             G0=G0,
             ISP=np.array([300, 350]),
             EPSILON=np.array([0.1, 0.08]),
             TOTAL_DELTA_V=8000,
             description="Should accept (56.25%/43.75% split)")
    
    # Test Set 4: Edge Cases
    print("\nTest Set 4: Edge Cases")
    print("-" * 40)
    
    # 4.1: Very low ISP
    run_test("4.1 - Low ISP",
             dv=np.array([4000, 4000]),
             G0=G0,
             ISP=np.array([200, 200]),
             EPSILON=np.array([0.1, 0.08]),
             TOTAL_DELTA_V=8000,
             description="Should handle poor efficiency")
    
    # 4.2: High structural mass
    run_test("4.2 - High Structural Mass",
             dv=np.array([4000, 4000]),
             G0=G0,
             ISP=np.array([300, 350]),
             EPSILON=np.array([0.2, 0.2]),
             TOTAL_DELTA_V=8000,
             description="Should handle mass penalties")
             
    # 4.3: Extremely high ISP (unrealistic)
    run_test("4.3 - Unrealistic ISP",
             dv=np.array([4000, 4000]),
             G0=G0,
             ISP=np.array([1000, 1000]),
             EPSILON=np.array([0.1, 0.08]),
             TOTAL_DELTA_V=8000,
             description="Should handle unrealistic efficiency")
             
    # 4.4: Near-zero structural mass
    run_test("4.4 - Near-Zero Structure",
             dv=np.array([4000, 4000]),
             G0=G0,
             ISP=np.array([300, 350]),
             EPSILON=np.array([0.01, 0.01]),
             TOTAL_DELTA_V=8000,
             description="Should handle extremely light structure")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()
