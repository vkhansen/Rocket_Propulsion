"""Base solver class for optimization."""
from abc import ABC, abstractmethod
import numpy as np
from ...utils.config import logger
from ..objective import calculate_stage_ratios, calculate_payload_fraction
from ..cache import OptimizationCache

class BaseSolver(ABC):
    """Base class for all optimization solvers."""
    
    def __init__(self, config, problem_params):
        """Initialize solver.
        
        Args:
            config: Configuration dictionary
            problem_params: Dictionary containing problem parameters
                - G0: Gravitational constant
                - ISP: List of specific impulse values
                - EPSILON: List of structural fraction values
                - TOTAL_DELTA_V: Total required delta-v
        """
        self.config = config
        self.G0 = problem_params['G0']
        self.ISP = np.asarray(problem_params['ISP'])
        self.EPSILON = np.asarray(problem_params['EPSILON'])
        self.TOTAL_DELTA_V = problem_params['TOTAL_DELTA_V']
        self.cache = OptimizationCache()
        
        # Get solver-specific configuration
        solver_name = self.__class__.__name__.lower().replace('solver', '')
        self.solver_config = self._get_solver_config(solver_name)
        
    def _get_solver_config(self, solver_name):
        """Get solver-specific configuration."""
        opt_config = self.config.get('optimization', {})
        solver_config = opt_config.get('solvers', {}).get(solver_name, {})
        constraints = opt_config.get('constraints', {})
        
        # Add common settings
        solver_config['constraints'] = constraints
        solver_config['penalty_coefficient'] = opt_config.get('penalty_coefficient', 1e3)
        
        return solver_config
        
    def calculate_fitness(self, dv):
        """Calculate fitness (payload fraction) for a solution."""
        # Check cache first
        cached_fitness = self.cache.get_cached_fitness(dv)
        if cached_fitness is not None:
            return cached_fitness
            
        # Calculate stage ratios and payload fraction
        stage_ratios, _ = calculate_stage_ratios(dv, self.G0, self.ISP, self.EPSILON)
        payload_fraction = calculate_payload_fraction(stage_ratios)
        
        # Cache result
        self.cache.add(dv, payload_fraction)
        
        return payload_fraction
        
    def enforce_constraints(self, dv):
        """Calculate constraint violation penalty."""
        # Get constraint parameters
        constraints = self.solver_config['constraints']
        stage_fractions = constraints.get('stage_fractions', {})
        first_stage = stage_fractions.get('first_stage', {})
        other_stages = stage_fractions.get('other_stages', {})
        
        # Get constraint values
        min_fraction_first = first_stage.get('min_fraction', 0.15)
        max_fraction_first = first_stage.get('max_fraction', 0.80)
        min_fraction_other = other_stages.get('min_fraction', 0.01)
        max_fraction_other = other_stages.get('max_fraction', 0.90)
        
        # Calculate total DV violation
        total_dv = np.sum(dv)
        total_dv_error = abs(total_dv - self.TOTAL_DELTA_V)
        
        # Calculate stage fraction violations
        stage_fractions = dv / total_dv if total_dv > 0 else np.zeros_like(dv)
        
        penalty = 0.0
        # First stage constraints
        if stage_fractions[0] < min_fraction_first:
            penalty += abs(stage_fractions[0] - min_fraction_first)
        if stage_fractions[0] > max_fraction_first:
            penalty += abs(stage_fractions[0] - max_fraction_first)
            
        # Other stages constraints
        for fraction in stage_fractions[1:]:
            if fraction < min_fraction_other:
                penalty += abs(fraction - min_fraction_other)
            if fraction > max_fraction_other:
                penalty += abs(fraction - max_fraction_other)
                
        return penalty + total_dv_error
        
    @abstractmethod
    def solve(self, initial_guess, bounds):
        """Solve the optimization problem.
        
        Args:
            initial_guess: Initial solution vector
            bounds: List of (min, max) bounds for each variable
            
        Returns:
            dict: Optimization results containing:
                - success: Whether optimization succeeded
                - message: Status message
                - payload_fraction: Best payload fraction found
                - stages: List of stage information
                - n_iterations: Number of iterations
                - n_function_evals: Number of function evaluations
        """
        pass
