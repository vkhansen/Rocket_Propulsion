"""SLSQP solver implementation."""
import numpy as np
from scipy.optimize import minimize
from .base_solver import BaseSolver
from ...utils.config import logger

class SLSQPSolver(BaseSolver):
    """SLSQP solver implementation."""
    
    def __init__(self, config, problem_params):
        """Initialize SLSQP solver."""
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
        """Solve using SLSQP algorithm."""
        try:
            # Get solver parameters
            max_iter = self.solver_specific.get('max_iterations', 1000)
            ftol = self.solver_specific.get('ftol', 1e-6)
            eps = self.solver_specific.get('eps', 1e-8)
            
            # Run optimization
            result = minimize(
                self.objective,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                options={
                    'maxiter': max_iter,
                    'ftol': ftol,
                    'eps': eps
                }
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
            logger.error(f"Error in SLSQP solver: {str(e)}")
            return {
                'success': False,
                'message': f"Error: {str(e)}",
                'payload_fraction': 0.0,
                'stages': [],
                'n_iterations': 0,
                'n_function_evals': 0
            }
