"""Differential Evolution solver implementation."""
import numpy as np
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
                    'message': f"DE optimization failed: {result.message}"
                }
            
        except Exception as e:
            logger.error(f"Error in DE solver: {str(e)}")
            return {
                'success': False,
                'message': f"Error in DE solver: {str(e)}"
            }
