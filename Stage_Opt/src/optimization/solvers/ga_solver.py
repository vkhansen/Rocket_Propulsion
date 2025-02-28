"""Genetic Algorithm solver implementation."""
import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from .base_solver import BaseSolver
from ...utils.config import logger

class RocketStageProblem(Problem):
    """Problem definition for pymoo GA."""
    
    def __init__(self, solver, n_var, bounds):
        super().__init__(
            n_var=n_var,
            n_obj=1,
            n_constr=0,
            xl=np.array([b[0] for b in bounds]),
            xu=np.array([b[1] for b in bounds])
        )
        self.solver = solver
        
    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate solutions."""
        f = np.array([self.solver.objective_with_penalty(xi) for xi in x])
        out["F"] = f

class GeneticAlgorithmSolver(BaseSolver):
    """Genetic Algorithm solver implementation."""
    
    def __init__(self, config, problem_params):
        """Initialize GA solver."""
        super().__init__(config, problem_params)
        self.solver_specific = self.solver_config.get('solver_specific', {})
        
    def solve(self, initial_guess, bounds):
        """Solve using Genetic Algorithm.
        
        Args:
            initial_guess: Initial solution guess
            bounds: List of (min, max) bounds for each variable
            
        Returns:
            dict: Optimization results
        """
        try:
            logger.info("Starting GA optimization...")
            
            # Get solver parameters
            pop_size = self.solver_specific.get('population_size', 100)
            n_gen = self.solver_specific.get('n_generations', 100)
            
            # Initialize problem
            problem = RocketStageProblem(self, len(initial_guess), bounds)
            
            # Initialize algorithm
            algorithm = GA(
                pop_size=pop_size,
                eliminate_duplicates=True
            )
            
            # Run optimization
            result = minimize(
                problem,
                algorithm,
                ('n_gen', n_gen),
                seed=42,
                verbose=False
            )
            
            # Process results
            if result.success:
                x = result.X
                stage_ratios, mass_ratios = self.calculate_stage_ratios(x)
                payload_fraction = self.calculate_fitness(x)
                
                return {
                    'success': True,
                    'x': x.tolist(),
                    'fun': float(result.F[0]),
                    'payload_fraction': float(payload_fraction),
                    'stage_ratios': stage_ratios.tolist(),
                    'mass_ratios': mass_ratios.tolist(),
                    'stages': self.create_stage_results(x, stage_ratios),
                    'n_iterations': result.algorithm.n_iter,
                    'n_function_evals': result.algorithm.evaluator.n_eval
                }
            else:
                return {
                    'success': False,
                    'message': "GA optimization failed to converge"
                }
            
        except Exception as e:
            logger.error(f"Error in GA solver: {str(e)}")
            return {
                'success': False,
                'message': f"Error in GA solver: {str(e)}"
            }
