"""Differential Evolution solver implementation."""
import numpy as np
from scipy.optimize import differential_evolution
from .base_solver import BaseSolver
from ...utils.config import logger

class DESolver(BaseSolver):
    """Differential Evolution solver implementation."""
    
    def __init__(self, config, problem_params):
        """Initialize DE solver."""
        super().__init__(config, problem_params)
        self.solver_specific = self.solver_config.get('solver_specific', {})
        
    def objective(self, x):
        """Objective function for optimization."""
        # Calculate constraint violation penalty
        penalty = self.enforce_constraints(x)
        
        # Calculate payload fraction (negative for minimization)
        payload_fraction = -self.calculate_fitness(x)
        
        # Apply penalty
        penalty_coeff = self.solver_config.get('penalty_coefficient', 1e3)
        return payload_fraction + penalty_coeff * penalty
        
    def solve(self, initial_guess, bounds):
        """Solve using Differential Evolution."""
        try:
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
                init='latinhypercube'
            )
            
            # Process results
            success = result.success
            message = result.message
            x = result.x
            n_iter = result.nit
            n_evals = result.nfev
            
            # Calculate final payload fraction
            payload_fraction = self.calculate_fitness(x)
            
            # Calculate stage information
            stage_ratios, stage_info = calculate_stage_ratios(
                x, self.G0, self.ISP, self.EPSILON
            )
            
            return {
                'success': success,
                'message': message,
                'payload_fraction': payload_fraction,
                'stages': stage_info,
                'n_iterations': n_iter,
                'n_function_evals': n_evals
            }
            
        except Exception as e:
            logger.error(f"Error in DE solver: {str(e)}")
            return {
                'success': False,
                'message': f"Error: {str(e)}",
                'payload_fraction': 0.0,
                'stages': [],
                'n_iterations': 0,
                'n_function_evals': 0
            }
