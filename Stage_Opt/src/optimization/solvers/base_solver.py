"""Base solver class for optimization."""
from abc import ABC, abstractmethod
import numpy as np
from ...utils.config import logger
from ..physics import calculate_stage_ratios, calculate_mass_ratios, calculate_payload_fraction
from ..cache import OptimizationCache

class BaseSolver(ABC):
    """Base class for all optimization solvers."""
    
    def __init__(self, config, problem_params):
        """Initialize solver."""
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
        
    @property
    def name(self):
        """Get solver name."""
        return self.__class__.__name__
        
    def create_stage_results(self, x, stage_ratios):
        """Create detailed results for each stage."""
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
        
    def enforce_constraints(self, x):
        """Enforce optimization constraints."""
        try:
            # Convert input to numpy array if needed
            x_arr = np.asarray(x, dtype=float)
            
            # Calculate total delta-v constraint violation
            total_dv = float(np.sum(x_arr))
            violation = abs(total_dv - self.TOTAL_DELTA_V)
            
            # Add penalties for negative values
            violation += np.sum(np.abs(np.minimum(x_arr, 0)))
            
            return float(violation)
            
        except Exception as e:
            logger.error(f"Error enforcing constraints: {e}")
            return float('inf')
    
    def process_results(self, x, success=True, message="", n_iterations=0, n_function_evals=0, time=0.0):
        """Process optimization results into a standardized format."""
        try:
            # Convert x to numpy array if needed
            x = np.asarray(x, dtype=float)
            
            # Calculate ratios and payload fraction
            stage_ratios, mass_ratios = self.calculate_stage_ratios(x)
            payload_fraction = calculate_payload_fraction(mass_ratios, self.EPSILON)
            
            # Create detailed stage results
            stages = self.create_stage_results(x, stage_ratios)
            
            return {
                'success': bool(success),
                'message': str(message),
                'method': self.name,
                'x': x.tolist(),
                'dv': x.tolist(),  # For backward compatibility
                'stages': stages,
                'stage_ratios': [float(r) for r in stage_ratios],
                'mass_ratios': [float(r) for r in mass_ratios],
                'payload_fraction': float(payload_fraction),
                'n_iterations': int(n_iterations),
                'n_function_evals': int(n_function_evals),
                'execution_time': float(time),
                'constraint_violation': float(self.enforce_constraints(x))
            }
            
        except Exception as e:
            logger.error(f"Error processing results: {str(e)}")
            return {
                'success': False,
                'message': f"Error processing results: {str(e)}",
                'method': self.name,
                'x': [],
                'dv': [],
                'stages': [],
                'stage_ratios': [],
                'mass_ratios': [],
                'payload_fraction': 0.0,
                'n_iterations': 0,
                'n_function_evals': 0,
                'execution_time': 0.0,
                'constraint_violation': float('inf')
            }
    
    def calculate_stage_ratios(self, x):
        """Calculate stage ratios for a solution."""
        try:
            x_arr = np.asarray(x, dtype=float)
            return calculate_stage_ratios(x_arr, self.G0, self.ISP, self.EPSILON)
        except Exception as e:
            logger.error(f"Error calculating stage ratios: {e}")
            return np.ones_like(x), np.ones_like(x)
    
    @abstractmethod
    def solve(self, initial_guess, bounds):
        """Solve optimization problem."""
        pass
