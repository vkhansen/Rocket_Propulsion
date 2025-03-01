"""Differential Evolution solver implementation."""
import numpy as np
import time
from scipy.optimize import differential_evolution
from ...utils.config import logger
from .base_solver import BaseSolver
from ..objective import objective_with_penalty

class DifferentialEvolutionSolver(BaseSolver):
    """Differential Evolution solver implementation."""
    
    def __init__(self, config, problem_params):
        """Initialize DE solver."""
        super().__init__(config, problem_params)
        self.solver_specific = self.solver_config.get('solver_specific', {})
        
    def objective(self, x):
        """Objective function for optimization."""
        return objective_with_penalty(
            dv=x,
            G0=self.G0,
            ISP=self.ISP,
            EPSILON=self.EPSILON,
            TOTAL_DELTA_V=self.TOTAL_DELTA_V
        )
        
    def solve(self, initial_guess, bounds):
        """Solve using Differential Evolution.
        
        Args:
            initial_guess: Initial solution guess
            bounds: List of (min, max) bounds for each variable
            
        Returns:
            dict: Optimization results
        """
        try:
            logger.info("Starting DE optimization...")
            start_time = time.time()
            
            # Get solver parameters
            max_iter = int(self.solver_specific.get('maxiter', 1000))
            pop_size = int(self.solver_specific.get('popsize', 15))
            mutation = self.solver_specific.get('mutation', (0.5, 1.0))
            recombination = float(self.solver_specific.get('recombination', 0.7))
            seed = int(self.solver_specific.get('seed', 42))
            strategy = str(self.solver_specific.get('strategy', 'best1bin'))
            
            # Run optimization
            result = differential_evolution(
                self.objective,
                bounds=bounds,
                maxiter=max_iter,
                popsize=pop_size,
                mutation=mutation,
                recombination=recombination,
                seed=seed,
                strategy=strategy,
                init='sobol'
            )
            
            execution_time = time.time() - start_time
            return self.process_results(
                x=result.x,
                success=result.success,
                message=result.message,
                n_iterations=result.nit,
                n_function_evals=result.nfev,
                time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Error in DE solver: {str(e)}")
            return self.process_results(
                x=initial_guess,
                success=False,
                message=str(e)
            )
