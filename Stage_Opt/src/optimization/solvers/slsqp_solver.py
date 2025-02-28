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
        return self.objective_with_penalty(x)
        
    def solve(self):
        """Solve using SLSQP."""
        try:
            logger.info("Starting SLSQP optimization...")
            
            # Get solver parameters
            maxiter = self.solver_specific.get('max_iterations', 100)
            ftol = self.solver_specific.get('ftol', 1e-6)
            
            # Run optimization
            result = minimize(
                self.objective,
                x0=self.initial_guess,
                method='SLSQP',
                bounds=self.bounds,
                options={
                    'maxiter': maxiter,
                    'ftol': ftol,
                    'disp': False
                }
            )
            
            # Process results
            if result.success:
                stage_ratios, mass_ratios = self.calculate_stage_ratios(result.x)
                payload_fraction = self.calculate_payload_fraction(stage_ratios)
                
                return {
                    'success': True,
                    'x': result.x.tolist(),
                    'fun': float(result.fun),
                    'payload_fraction': payload_fraction,
                    'stage_ratios': stage_ratios.tolist(),
                    'mass_ratios': mass_ratios.tolist(),
                    'stages': self.create_stage_results(result.x, stage_ratios),
                    'n_iterations': result.nit,
                    'n_function_evals': result.nfev
                }
            else:
                return {
                    'success': False,
                    'message': f"SLSQP optimization failed: {result.message}"
                }
            
        except Exception as e:
            logger.error(f"Error in SLSQP solver: {str(e)}")
            return {
                'success': False,
                'message': f"Error in SLSQP solver: {str(e)}"
            }
