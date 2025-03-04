"""Test script for AdaptiveGeneticAlgorithmSolver."""
import numpy as np
from src.optimization.solvers.adaptive_ga_solver import AdaptiveGeneticAlgorithmSolver
from src.utils.config import logger
from src.optimization.physics import calculate_stage_ratios, calculate_payload_fraction
import logging
import time

# Configure logging
logger.setLevel(logging.INFO)

def run_test_case(test_name, G0, ISP, EPSILON, TOTAL_DELTA_V, pop_size=50, n_generations=50):
    """Run a test case with the given parameters."""
    print(f"\n=== Running Test Case: {test_name} ===")
    
    n_stages = len(ISP)
    bounds = [(0, TOTAL_DELTA_V) for _ in range(n_stages)]
    
    # Initial guess - equal distribution
    initial_guess = np.array([TOTAL_DELTA_V / n_stages] * n_stages)
    
    # Create solver with specific configuration
    config = {
        'solver_specific': {
            'population_size': pop_size,
            'n_generations': n_generations,
            'initial_mutation_rate': 0.2,
            'initial_crossover_rate': 0.8,
            'tournament_size': 3,
            'max_projection_iterations': 20
        }
    }
    
    solver = AdaptiveGeneticAlgorithmSolver(
        G0=G0, 
        ISP=ISP, 
        EPSILON=EPSILON, 
        TOTAL_DELTA_V=TOTAL_DELTA_V, 
        bounds=bounds,
        config=config
    )
    
    # Run solver and time it
    start_time = time.time()
    print(f"Running {solver.name}...")
    results = solver.solve(initial_guess, bounds)
    elapsed_time = time.time() - start_time
    
    # Print results
    print(f"Success: {results['success']}")
    print(f"Message: {results['message']}")
    print(f"Payload fraction: {results['payload_fraction']}")
    print(f"Stages: {results['stages']}")
    print(f"Iterations: {results['execution_metrics']['iterations']}")
    print(f"Is feasible: {results['success']}")
    print(f"Violation: {results['constraint_violation']}")
    print(f"Execution time: {elapsed_time:.2f} seconds")
    
    # Store the parameters with the results for verification
    results['G0'] = G0
    results['ISP'] = ISP
    results['epsilon_values'] = EPSILON
    
    return results

def main():
    """Run tests of the AdaptiveGeneticAlgorithmSolver with different configurations."""
    
    # Test case 1: Basic two-stage rocket (standard test)
    test1_results = run_test_case(
        test_name="Two-Stage Rocket (Standard)",
        G0=9.81,  # m/s^2
        ISP=[300, 350],  # s
        EPSILON=[0.1, 0.05],  # structural coefficients
        TOTAL_DELTA_V=9300,  # m/s
        pop_size=50,
        n_generations=50
    )
    
    # Test case 2: Three-stage rocket
    test2_results = run_test_case(
        test_name="Three-Stage Rocket",
        G0=9.81,  # m/s^2
        ISP=[280, 320, 450],  # s
        EPSILON=[0.12, 0.08, 0.04],  # structural coefficients
        TOTAL_DELTA_V=9500,  # m/s
        pop_size=50,
        n_generations=50
    )
    
    # Test case 3: Four-stage rocket with higher ISP
    test3_results = run_test_case(
        test_name="Four-Stage High-ISP Rocket",
        G0=9.81,  # m/s^2
        ISP=[290, 340, 410, 465],  # s
        EPSILON=[0.13, 0.09, 0.06, 0.03],  # structural coefficients
        TOTAL_DELTA_V=11000,  # m/s
        pop_size=50,
        n_generations=50
    )
    
    # Test case 4: Single-stage rocket (edge case)
    test4_results = run_test_case(
        test_name="Single-Stage Rocket (Edge Case)",
        G0=9.81,  # m/s^2
        ISP=[320],  # s
        EPSILON=[0.1],  # structural coefficients
        TOTAL_DELTA_V=4500,  # m/s
        pop_size=30,
        n_generations=30
    )
    
    # Compare payload fractions
    print("\n=== Summary of Test Results ===")
    print(f"Two-Stage Rocket: Payload Fraction = {test1_results['payload_fraction']}")
    print(f"Three-Stage Rocket: Payload Fraction = {test2_results['payload_fraction']}")
    print(f"Four-Stage Rocket: Payload Fraction = {test3_results['payload_fraction']}")
    print(f"Single-Stage Rocket: Payload Fraction = {test4_results['payload_fraction']}")
    
    # Verify the mathematical conventions are followed
    # λ (lambda) = mf/m0 (stage ratio)
    # μ (mu) = 1/(λ*(1-ε) + ε) (mass ratio)
    # Payload fraction = Π(1/μ_i)
    print("\n=== Verification of Mathematical Conventions ===")
    test_cases = [
        ("Two-Stage", test1_results),
        ("Three-Stage", test2_results),
        ("Four-Stage", test3_results),
        ("Single-Stage", test4_results)
    ]
    
    for test_name, results in test_cases:
        print(f"\n{test_name} Rocket Verification:")
        
        # Only verify if the solution is feasible and has stages
        if results['success'] and results['stages']:
            # Extract delta-v values and parameters
            delta_v_values = np.array([stage['delta_v'] for stage in results['stages']])
            G0 = results['G0']
            ISP = np.array(results['ISP'])
            EPSILON = np.array(results['epsilon_values'])
            
            print(f"  Delta-V values: {delta_v_values}")
            
            # Calculate stage ratios and mass ratios using the physics module
            stage_ratios, mass_ratios = calculate_stage_ratios(
                dv=delta_v_values,
                G0=G0,
                ISP=ISP,
                EPSILON=EPSILON
            )
            
            print(f"  Lambda values (stage ratios): {stage_ratios}")
            print(f"  Epsilon values: {EPSILON}")
            print(f"  Calculated mass ratios (μ): {mass_ratios}")
            
            # Calculate payload fraction using the physics module
            calculated_payload = calculate_payload_fraction(mass_ratios)
            
            print(f"  Calculated payload fraction: {calculated_payload}")
            print(f"  Reported payload fraction: {results['payload_fraction']}")
            print(f"  Match? {abs(calculated_payload - results['payload_fraction']) < 1e-6}")
            
            # Verify lambda values match what's in the results
            lambda_values_match = all(
                abs(sr - stage['Lambda']) < 1e-6 
                for sr, stage in zip(stage_ratios, results['stages'])
            )
            print(f"  Lambda values match? {lambda_values_match}")
            
            # Calculate total delta-v
            total_dv = sum(stage['delta_v'] for stage in results['stages'])
            print(f"  Total Delta-V: {total_dv}")
        else:
            print(f"  Solution is not feasible or has no stages. Cannot verify.")
            print(f"  Success: {results['success']}")
            print(f"  Constraint violation: {results['constraint_violation']}")
    
    return 0

if __name__ == "__main__":
    exit(main())
