"""Basin Hopping solver implementation."""
import numpy as np
from scipy.optimize import basinhopping
from .base_solver import BaseSolver
from ...utils.config import logger

class BasinHoppingOptimizer(BaseSolver):
    """Basin Hopping solver implementation."""
    
    def __init__(self, config, problem_params):
        """Initialize Basin Hopping solver."""
        super().__init__(config, problem_params)
        self.solver_specific = self.solver_config.get('solver_specific', {})
        
    def objective(self, x):
        """Objective function for optimization."""
        return -self.objective_with_penalty(x)  # Negative because Basin Hopping minimizes
        
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
                minimizer_kwargs=minimizer_kwargs,
                seed=42
            )
            
            if result.success:
                x = result.x
                stage_ratios, mass_ratios = self.calculate_stage_ratios(x)
                payload_fraction = self.calculate_fitness(x)
                
                return {
                    'success': True,
                    'x': x.tolist(),
                    'fun': float(-result.fun),  # Convert back to maximization
                    'payload_fraction': float(payload_fraction),
                    'stage_ratios': stage_ratios.tolist(),
                    'mass_ratios': mass_ratios.tolist(),
                    'stages': self.create_stage_results(x, stage_ratios),
                    'n_iterations': result.nit,
                    'n_function_evals': result.nfev
                }
            else:
                return {
                    'success': False,
                    'message': f"Basin Hopping optimization failed: {result.message}"
                }
            
        except Exception as e:
            logger.error(f"Error in Basin Hopping solver: {str(e)}")
            return {
                'success': False,
                'message': f"Error in Basin Hopping solver: {str(e)}"
            }
