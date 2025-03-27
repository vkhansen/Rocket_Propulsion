from src.optimization.solvers.ga_solver import GeneticAlgorithmSolver
import numpy as np

def main():
    solver = GeneticAlgorithmSolver(
        G0=9.81, 
        ISP=[312, 348], 
        EPSILON=[0.07, 0.08], 
        TOTAL_DELTA_V=9300, 
        bounds=[(0, 9300), (0, 9300)], 
        config=None, 
        max_generations=5
    )
    
    result = solver.solve()
    
    print(f"Success: {result['success']}")
    
    # Check if there are stages in the result
    if 'stages' in result and result['stages']:
        # Extract the solution from stages
        solution = [stage['delta_v'] for stage in result['stages']]
        print(f"Solution: {solution}")
    else:
        print("No valid solution found in the result")
    
    print(f"Payload Fraction: {result['payload_fraction']}")
    print(f"Iterations: {result['execution_metrics']['iterations']}")
    print(f"Is feasible: {result['constraint_violation'] <= 1e-6}")
    print(f"Violation: {result['constraint_violation']}")

if __name__ == "__main__":
    main()
