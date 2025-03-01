"""Differential Evolution solver implementation."""
import numpy as np
import time
from scipy.optimize import differential_evolution
from .base_solver import BaseSolver
from ...utils.config import logger

class DifferentialEvolutionSolver(BaseSolver):
    """Differential Evolution solver implementation."""
    
    def __init__(self, config, problem_params):
        """Initialize DE solver."""
        super().__init__(config, problem_params)
        self.solver_specific = self.solver_config.get('solver_specific', {})
        
    def objective(self, x):
        """Objective function for optimization."""
        return -self.objective_with_penalty(x)  # Negative because DE minimizes
        
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
            pop_size = self.solver_specific.get('population_size', 20)
            max_iter = self.solver_specific.get('max_iterations', 1000)
            strategy = self.solver_specific.get('strategy', 'best1bin')
            mutation = self.solver_specific.get('mutation', [0.5, 1.0])
            recombination = self.solver_specific.get('recombination', 0.7)
            
            # Run optimization
            result = differential_evolution(
                self.objective,
                bounds,
                strategy=strategy,
                maxiter=max_iter,
                popsize=pop_size,
                mutation=mutation,
                recombination=recombination,
                init='latinhypercube',
                seed=42,
                workers=1  # For reproducibility
            )
            
            execution_time = time.time() - start_time
            return self.process_results(
                result.x,
                success=result.success,
                message=result.message,
                n_iter=result.nit,
                n_evals=result.nfev,
                time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Error in DE solver: {str(e)}")
            return self.process_results(
                np.zeros_like(initial_guess),
                success=False,
                message=f"Error in DE solver: {str(e)}"
            )
