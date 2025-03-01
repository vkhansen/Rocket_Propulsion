"""Basin Hopping solver implementation."""
import numpy as np
from scipy.optimize import basinhopping
from ...utils.config import logger
from .base_solver import BaseSolver
from ..objective import objective_with_penalty

class BasinHoppingOptimizer(BaseSolver):
    """Basin Hopping solver implementation."""
    
    def __init__(self, config, problem_params):
        """Initialize Basin Hopping solver."""
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
        """Solve using Basin Hopping.
        
        Args:
            initial_guess: Initial solution guess
            bounds: List of (min, max) bounds for each variable
            
        Returns:
            dict: Optimization results
        """
        try:
            logger.info("Starting Basin Hopping optimization...")
            
            # Get solver parameters
            niter = int(self.solver_specific.get('niter', 100))
            T = float(self.solver_specific.get('T', 1.0))
            stepsize = float(self.solver_specific.get('stepsize', 0.5))
            maxiter = int(self.solver_specific.get('maxiter', 1000))
            ftol = float(self.solver_specific.get('ftol', 1e-6))
            
            # Create bounds constraint
            bounds_list = [(low, high) for low, high in bounds]
            
            class BoundsConstraint:
                """Bounds constraint for Basin Hopping."""
                def __init__(self, bounds):
                    self.bounds = bounds
                    
                def __call__(self, **kwargs):
                    """Check if x is within bounds."""
                    x = kwargs.get('x_new')
                    return np.all((x >= self.bounds[:, 0]) & (x <= self.bounds[:, 1]))
            
            # Run optimization
            minimizer_kwargs = {
                'method': 'SLSQP',
                'bounds': bounds_list,
                'options': {
                    'maxiter': maxiter,
                    'ftol': ftol
                }
            }
            
            result = basinhopping(
                self.objective,
                x0=initial_guess,
                niter=niter,
                T=T,
                stepsize=stepsize,
                minimizer_kwargs=minimizer_kwargs,
                accept_test=BoundsConstraint(np.array(bounds))
            )
            
            return self.process_results(
                x=result.x,
                success=result.success,
                message=result.message,
                n_iterations=result.nit,
                n_function_evals=result.nfev,
                time=0.0  # Time not tracked by scipy
            )
            
        except Exception as e:
            logger.error(f"Error in Basin Hopping solver: {str(e)}")
            return self.process_results(
                x=initial_guess,
                success=False,
                message=str(e)
            )
