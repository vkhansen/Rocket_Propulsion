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
    print(f"Solution: {result['x']}")
    print(f"Fitness: {result['fun']}")
    print(f"Iterations: {result['nit']}")
    print(f"Is feasible: {result['is_feasible']}")
    print(f"Violation: {result['violation']}")

if __name__ == "__main__":
    main()
