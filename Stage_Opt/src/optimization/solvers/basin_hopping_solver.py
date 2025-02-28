"""Basin Hopping solver implementation."""
import numpy as np
from scipy.optimize import basinhopping
from .base_solver import BaseSolver
from utils.config import logger

class BasinHoppingSolver(BaseSolver):
    """Basin Hopping solver implementation."""
    
    def __init__(self, config, problem_params):
        """Initialize Basin Hopping solver."""
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
        """Solve using Basin Hopping."""
        try:
            # Get solver parameters
            n_iter = self.solver_specific.get('n_iterations', 100)
            T = self.solver_specific.get('temperature', 1.0)
            stepsize = self.solver_specific.get('stepsize', 0.5)
            interval = self.solver_specific.get('interval', 50)
            
            # Create minimizer kwargs for local optimization
            minimizer_kwargs = {
                'method': 'L-BFGS-B',
                'bounds': bounds
            }
            
            # Run optimization
            result = basinhopping(
                self.objective,
                initial_guess,
                niter=n_iter,
                T=T,
                stepsize=stepsize,
                interval=interval,
                minimizer_kwargs=minimizer_kwargs
            )
            
            # Process results
            success = result.lowest_optimization_result.success
            message = result.message
            x = result.x
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
                'n_function_evals': n_evals,
                'n_local_mins': len(result.lowest_optimization_result)
            }
            
        except Exception as e:
            logger.error(f"Error in Basin Hopping solver: {str(e)}")
            return {
                'success': False,
                'message': f"Error: {str(e)}",
                'payload_fraction': 0.0,
                'stages': [],
                'n_iterations': 0,
                'n_function_evals': 0
            }
