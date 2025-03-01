"""Base solver class for optimization."""
from abc import ABC, abstractmethod
from typing import Dict
import numpy as np

from ...utils.config import logger
from ..cache import OptimizationCache
from ..physics import calculate_stage_ratios, calculate_payload_fraction
from ..objective import objective_with_penalty

class BaseSolver(ABC):
    """Base class for all optimization solvers."""
    
    def __init__(self, config: Dict, problem_params: Dict):
        """Initialize solver with configuration and problem parameters.
        
        Args:
            config: Dictionary containing solver configuration
            problem_params: Dictionary containing problem parameters
        """
        self.config = config if isinstance(config, dict) else {}
        self.solver_config = self.config.get('optimization', {}).get('solver', {})
        
        # Problem parameters - ensure all values are Python floats
        if not isinstance(problem_params, dict):
            problem_params = {}
            
        self.G0 = float(problem_params.get('G0', 9.81))
        stages = problem_params.get('stages', [])
        self.ISP = np.array([float(stage.get('ISP', 0.0)) for stage in stages])
        self.EPSILON = np.array([float(stage.get('EPSILON', 0.0)) for stage in stages])
        self.TOTAL_DELTA_V = float(problem_params.get('TOTAL_DELTA_V', 0.0))
        
        # Initialize cache
        self.cache = OptimizationCache()
        
        # Initialize name
        self.name = self.__class__.__name__
        
    def objective_with_penalty(self, x):
        """Calculate objective value and constraints.
        
        Args:
            x: Input vector (delta-v values)
            
        Returns:
            tuple: (objective_value, dv_constraint, physical_constraint)
        """
        return objective_with_penalty(
            dv=x,
            G0=self.G0,
            ISP=self.ISP,
            EPSILON=self.EPSILON,
            TOTAL_DELTA_V=self.TOTAL_DELTA_V
        )
        
    def calculate_stage_ratios(self, x: np.ndarray):
        """Calculate stage ratios for a solution.
        
        Args:
            x: Array of delta-v values
            
        Returns:
            Tuple of (stage_ratios, mass_ratios)
        """
        try:
            # Ensure x is a 1D array matching the number of stages
            x = np.asarray(x, dtype=float).reshape(-1)
            if x.size != len(self.ISP):
                raise ValueError(f"Expected {len(self.ISP)} stages, got {x.size}")
                
            return calculate_stage_ratios(
                dv=x,
                G0=self.G0,
                ISP=self.ISP,
                EPSILON=self.EPSILON
            )
        except Exception as e:
            logger.error(f"Error calculating stage ratios: {str(e)}")
            return np.zeros_like(self.ISP), np.zeros_like(self.ISP)

    def enforce_constraints(self, x: np.ndarray):
        """Enforce optimization constraints.
        
        Args:
            x: Array of delta-v values
            
        Returns:
            Total constraint violation value
        """
        try:
            # Calculate constraint violations
            _, dv_constraint, physical_constraint = self.objective_with_penalty(x)
            return float(abs(dv_constraint) + abs(physical_constraint))
            
        except Exception as e:
            logger.error(f"Error enforcing constraints: {str(e)}")
            return float('inf')
            
    def process_results(self, x: np.ndarray, success: bool = True, message: str = "", 
                       n_iterations: int = 0, n_function_evals: int = 0, 
                       time: float = 0.0) -> Dict:
        """Process optimization results into a standardized format.
        
        Args:
            x: Solution vector (delta-v values)
            success: Whether optimization succeeded
            message: Status message from optimizer
            n_iterations: Number of iterations performed
            n_function_evals: Number of function evaluations
            time: Execution time in seconds
            
        Returns:
            Dictionary containing standardized optimization results
        """
        try:
            # Convert x to numpy array and validate
            x = np.asarray(x, dtype=float)
            if x.size == 0 or not np.all(np.isfinite(x)):
                raise ValueError("Invalid solution vector")
            
            # Calculate ratios and payload fraction
            stage_ratios, mass_ratios = self.calculate_stage_ratios(x)
            payload_fraction = calculate_payload_fraction(mass_ratios, self.EPSILON)
            
            # Calculate constraint violations
            _, dv_constraint, physical_constraint = self.objective_with_penalty(x)
            
            return {
                'solver': self.name,
                'success': success,
                'message': message,
                'x': x.tolist(),
                'stage_ratios': stage_ratios.tolist(),
                'mass_ratios': mass_ratios.tolist(),
                'payload_fraction': float(payload_fraction),
                'dv_constraint': float(dv_constraint),
                'physical_constraint': float(physical_constraint),
                'n_iterations': n_iterations,
                'n_function_evals': n_function_evals,
                'time': time
            }
            
        except Exception as e:
            logger.error(f"Error processing results: {str(e)}")
            return {
                'solver': self.name,
                'success': False,
                'message': f"Failed to process results: {str(e)}",
                'x': x.tolist() if isinstance(x, np.ndarray) else x,
                'stage_ratios': [],
                'mass_ratios': [],
                'payload_fraction': 0.0,
                'dv_constraint': float('inf'),
                'physical_constraint': float('inf'),
                'n_iterations': n_iterations,
                'n_function_evals': n_function_evals,
                'time': time
            }
            
    @abstractmethod
    def solve(self, initial_guess, bounds):
        """Solve the optimization problem.
        
        Args:
            initial_guess: Initial solution guess
            bounds: List of (min, max) bounds for each variable
            
        Returns:
            Dictionary containing optimization results
        """
        pass
