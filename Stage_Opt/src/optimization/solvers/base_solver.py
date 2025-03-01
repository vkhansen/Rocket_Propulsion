"""Base solver class for optimization."""
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Union, Optional
from ...utils.config import logger
from ..physics import calculate_stage_ratios, calculate_mass_ratios, calculate_payload_fraction
from ..cache import OptimizationCache

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
        
        # Validate initialization
        if len(self.ISP) != len(self.EPSILON):
            raise ValueError("Number of ISP values must match number of EPSILON values")
        if len(self.ISP) == 0:
            raise ValueError("At least one stage must be defined")
        if self.TOTAL_DELTA_V <= 0:
            raise ValueError("Total delta-v must be positive")
        
    @property
    def name(self) -> str:
        """Get solver name."""
        return self.__class__.__name__
        
    def create_stage_results(self, x: np.ndarray, stage_ratios: np.ndarray) -> List[Dict]:
        """Create detailed results for each stage.
        
        Args:
            x: Array of delta-v values
            stage_ratios: Array of stage ratios
            
        Returns:
            List of dictionaries containing stage-specific results
        """
        try:
            stages = []
            for i, (dv_i, lambda_i) in enumerate(zip(x, stage_ratios)):
                stage = {
                    'stage': i + 1,
                    'delta_v': float(dv_i),
                    'Lambda': float(lambda_i),
                    'ISP': float(self.ISP[i]),
                    'EPSILON': float(self.EPSILON[i])
                }
                stages.append(stage)
            return stages
        except Exception as e:
            logger.error(f"Error creating stage results: {e}")
            return []
        
    def enforce_constraints(self, x: np.ndarray) -> float:
        """Enforce optimization constraints.
        
        Args:
            x: Array of delta-v values
            
        Returns:
            Total constraint violation value
        """
        try:
            # Convert input to numpy array if needed
            x_arr = np.asarray(x, dtype=float)
            
            # Calculate total delta-v constraint violation
            total_dv = float(np.sum(x_arr))
            violation = abs(total_dv - self.TOTAL_DELTA_V)
            
            # Add penalties for negative values and physical constraints
            violation += np.sum(np.abs(np.minimum(x_arr, 0)))
            
            # Add penalties for unrealistic stage ratios
            stage_ratios, _ = self.calculate_stage_ratios(x_arr)
            min_ratio, max_ratio = 0.1, 0.9
            ratio_violations = np.sum(
                np.where(stage_ratios < min_ratio, min_ratio - stage_ratios, 0) +
                np.where(stage_ratios > max_ratio, stage_ratios - max_ratio, 0)
            )
            violation += ratio_violations
            
            return float(violation)
            
        except Exception as e:
            logger.error(f"Error enforcing constraints: {e}")
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
            
            # Validate results
            if not np.all(np.isfinite(stage_ratios)) or not np.all(np.isfinite(mass_ratios)):
                raise ValueError("Invalid stage or mass ratios")
            
            # Create detailed stage results
            stages = self.create_stage_results(x, stage_ratios)
            
            # Calculate constraint violation
            constraint_violation = self.enforce_constraints(x)
            
            return {
                'success': bool(success),
                'message': str(message),
                'method': self.name,
                'dv': x.tolist(),
                'stage_ratios': stage_ratios.tolist(),
                'mass_ratios': mass_ratios.tolist(),
                'payload_fraction': float(payload_fraction),
                'constraint_violation': float(constraint_violation),
                'execution_time': float(time),
                'n_iterations': int(n_iterations),
                'n_function_evals': int(n_function_evals),
                'stages': stages,
                'timestamp': None  # Will be set by parallel solver
            }
            
        except Exception as e:
            logger.error(f"Error processing results: {str(e)}")
            return {
                'success': False,
                'message': f"Error processing results: {str(e)}",
                'method': self.name,
                'dv': [],
                'stage_ratios': [],
                'mass_ratios': [],
                'payload_fraction': 0.0,
                'constraint_violation': float('inf'),
                'execution_time': float(time),
                'n_iterations': int(n_iterations),
                'n_function_evals': int(n_function_evals),
                'stages': [],
                'timestamp': None
            }
    
    def calculate_stage_ratios(self, x: np.ndarray) -> tuple:
        """Calculate stage ratios for a solution.
        
        Args:
            x: Array of delta-v values
            
        Returns:
            Tuple of (stage_ratios, mass_ratios)
        """
        try:
            x_arr = np.asarray(x, dtype=float)
            if not np.all(np.isfinite(x_arr)):
                raise ValueError("Invalid delta-v values")
            return calculate_stage_ratios(x_arr, self.G0, self.ISP, self.EPSILON)
        except Exception as e:
            logger.error(f"Error calculating stage ratios: {e}")
            return np.ones_like(x), np.ones_like(x)
    
    @abstractmethod
    def solve(self, initial_guess: np.ndarray, bounds: List[tuple]) -> Dict:
        """Solve optimization problem.
        
        Args:
            initial_guess: Initial solution vector
            bounds: List of (min, max) tuples for each variable
            
        Returns:
            Dictionary containing optimization results
        """
        pass
