"""SLSQP solver implementation."""
import numpy as np
from scipy.optimize import minimize
from ...utils.config import logger
from .base_solver import BaseSolver
from ..objective import objective_with_penalty

class SLSQPSolver(BaseSolver):
    """SLSQP solver implementation."""
    
    def __init__(self, config, problem_params):
        """Initialize SLSQP solver."""
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
        """Solve using SLSQP."""
        try:
            logger.info("Starting SLSQP optimization...")
            
            # Get solver parameters
            maxiter = self.solver_specific.get('max_iterations', 100)
            ftol = self.solver_specific.get('ftol', 1e-6)
            
            # Run optimization
            result = minimize(
                self.objective,
                x0=initial_guess,
                method='SLSQP',
                bounds=bounds,
                options={
                    'maxiter': maxiter,
                    'ftol': ftol,
                    'disp': False
                }
            )
            
            return self.process_results(
                x=result.x,
                success=result.success,
                message=result.message if not result.success else "",
                n_iterations=result.nit,
                n_function_evals=result.nfev,
                time=0.0  # Time not tracked by scipy
            )
            
        except Exception as e:
            logger.error(f"Error in SLSQP solver: {str(e)}")
            return self.process_results(
                x=initial_guess,
                success=False,
                message=str(e)
            )
