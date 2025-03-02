"""SLSQP solver implementation."""
import numpy as np
from scipy.optimize import minimize
from ...utils.config import logger
from .base_solver import BaseSolver
from ..objective import objective_with_penalty
import time

class SLSQPSolver(BaseSolver):
    """SLSQP solver implementation."""
    
    def __init__(self, G0, ISP, EPSILON, TOTAL_DELTA_V, bounds, max_iterations=100, ftol=1e-6):
        """Initialize SLSQP solver with problem parameters."""
        super().__init__(G0, ISP, EPSILON, TOTAL_DELTA_V, bounds)
        self.max_iterations = max_iterations
        self.ftol = ftol
        
    def objective(self, x):
        """Objective function for optimization."""
        return objective_with_penalty(
            dv=x,
            G0=self.G0,
            ISP=self.ISP,
            EPSILON=self.EPSILON,
            TOTAL_DELTA_V=self.TOTAL_DELTA_V,
            return_tuple=False  # Get scalar for SLSQP
        )
        
    def solve(self, initial_guess, bounds):
        """Solve using SLSQP."""
        try:
            logger.info("Starting SLSQP optimization...")
            
            # Run optimization
            start_time = time.time()
            result = minimize(
                self.objective,
                x0=initial_guess,
                method='SLSQP',
                bounds=bounds,
                options={
                    'maxiter': self.max_iterations,
                    'ftol': self.ftol,
                    'disp': False
                }
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
            logger.error(f"Error in SLSQP solver: {str(e)}")
            return self.process_results(
                x=initial_guess,
                success=False,
                message=str(e),
                n_iterations=0,
                n_function_evals=0,
                time=0.0
            )
