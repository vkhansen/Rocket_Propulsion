import time
import numpy as np
from scipy.optimize import basinhopping
from ...utils.config import logger
from .base_solver import BaseSolver
from ..objective import objective_with_penalty

class BasinHoppingOptimizer(BaseSolver):
    """Basin Hopping optimization solver implementation."""
    
    def __init__(self, G0, ISP, EPSILON, TOTAL_DELTA_V, bounds, niter=100, T=1.0, stepsize=0.5, minimizer_options=None):
        """Initialize Basin Hopping optimizer with direct problem parameters and BH-specific settings.
        
        Args:
            G0: Gravitational constant
            ISP: List of specific impulse values for each stage
            EPSILON: List of structural coefficients for each stage
            TOTAL_DELTA_V: Required total delta-v
            bounds: List of (min, max) bounds for each variable
            niter: Number of basin hopping iterations
            T: Temperature parameter for BH
            stepsize: Step size for local minimizer
            minimizer_options: Dictionary of options for the local minimizer
        """
        super().__init__(G0, ISP, EPSILON, TOTAL_DELTA_V, bounds)
        self.niter = niter
        self.T = T
        self.stepsize = stepsize
        self.minimizer_options = minimizer_options if minimizer_options is not None else {}
        
        logger.debug(
            f"Initialized {self.name} with parameters: niter={niter}, T={T}, stepsize={stepsize}, \
            minimizer_options={self.minimizer_options}"
        )
        
    def objective(self, x):
        """Objective function for Basin Hopping optimization."""
        return objective_with_penalty(
            dv=x,
            G0=self.G0,
            ISP=self.ISP,
            EPSILON=self.EPSILON,
            TOTAL_DELTA_V=self.TOTAL_DELTA_V,
            return_tuple=False
        )
        
    def solve(self, initial_guess, bounds):
        """Solve using Basin Hopping optimization."""
        try:
            logger.info("Starting Basin Hopping optimization...")
            start_time = time.time()
            
            minimizer_kwargs = {'method': 'L-BFGS-B', 'bounds': bounds}
            minimizer_kwargs.update(self.minimizer_options)
            
            result = basinhopping(
                self.objective,
                x0=initial_guess,
                minimizer_kwargs=minimizer_kwargs,
                niter=self.niter,
                T=self.T,
                stepsize=self.stepsize,
                disp=False
            )
            
            execution_time = time.time() - start_time
            
            return self.process_results(
                x=result.x,
                success=result.lowest_optimization_result.success,
                message=result.lowest_optimization_result.message,
                n_iterations=self.niter,
                n_function_evals=0,
                time=execution_time
            )
        except Exception as e:
            logger.error(f"Error in Basin Hopping optimizer: {str(e)}")
            return self.process_results(
                x=initial_guess,
                success=False,
                message=str(e),
                n_iterations=0,
                n_function_evals=0,
                time=0.0
            )
