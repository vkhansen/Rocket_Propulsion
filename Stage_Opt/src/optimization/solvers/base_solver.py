"""Base solver class for optimization."""
from abc import ABC, abstractmethod
import numpy as np
from ...utils.config import logger
from ..physics import calculate_stage_ratios, calculate_mass_ratios, calculate_payload_fraction
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
        solver_config = config.get('solver', {}) if isinstance(config, dict) else {}
        self.solver_config = solver_config
        
        # Problem parameters - ensure all values are Python floats
        if not isinstance(problem_params, dict):
            problem_params = {}
            
        self.G0 = float(problem_params.get('G0', 9.81))
        stages = problem_params.get('stages', [])
        self.ISP = [float(stage.get('ISP', 0.0)) for stage in stages]
        self.EPSILON = [float(stage.get('EPSILON', 0.0)) for stage in stages]
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
            x_arr = np.asarray(x)
            
            # Calculate total delta-v constraint violation
            total_dv = float(np.sum(x_arr))
            violation = abs(total_dv - self.TOTAL_DELTA_V)
            
            # Add penalties for negative values
            violation += np.sum(np.abs(np.minimum(x_arr, 0)))
            
            return float(violation)
            
        except Exception as e:
            logger.error(f"Error enforcing constraints: {e}")
            return float('inf')
    
    def calculate_stage_ratios(self, x):
        """Calculate stage ratios for a solution."""
        try:
            x_arr = np.asarray(x)
            return calculate_stage_ratios(x_arr, self.G0, self.ISP, self.EPSILON)
        except Exception as e:
            logger.error(f"Error calculating stage ratios: {e}")
            return np.ones_like(x), np.ones_like(x)
    
    def calculate_delta_v(self, stage_ratios):
        """Calculate delta-V for each stage.
        
        Args:
            stage_ratios (np.ndarray): Stage mass ratios (lambda)
            
        Returns:
            np.ndarray: Delta-V values for each stage
        """
        stage_ratios_arr = np.asarray(stage_ratios)
        return np.array([
            self.G0 * isp * np.log(ratio)
            for isp, ratio in zip(self.ISP, stage_ratios_arr)
        ])
        
    def calculate_fitness(self, x):
        """Calculate fitness (payload fraction) for a solution."""
        try:
            x_arr = np.asarray(x)
            
            # Check cache first
            cached_result = self.cache.get(tuple(x_arr.tolist()))
            if cached_result is not None:
                return float(cached_result)
            
            # Calculate mass ratios and payload fraction
            mass_ratios = calculate_mass_ratios(x_arr, self.ISP, self.EPSILON, self.G0)
            payload_fraction = calculate_payload_fraction(mass_ratios, self.EPSILON)
            
            # Cache result
            self.cache.set(tuple(x_arr.tolist()), float(payload_fraction))
            
            return float(payload_fraction)
            
        except Exception as e:
            logger.error(f"Error calculating fitness: {e}")
            return 0.0
    
    def objective_with_penalty(self, x):
        """Calculate objective with penalty for constraint violation."""
        try:
            # Calculate payload fraction (negative for minimization)
            payload_fraction = -self.calculate_fitness(x)
            
            # Calculate constraint violation penalty
            penalty = self.enforce_constraints(x)
            penalty_coeff = float(self.solver_config.get('penalty_coefficient', 1e3))
            
            return float(payload_fraction + penalty_coeff * penalty)
            
        except Exception as e:
            logger.error(f"Error in objective calculation: {e}")
            return float(1e6)
    
    def process_results(self, x, success, message="", n_iter=0, n_evals=0, time=0.0):
        """Process optimization results into a standardized format.
        
        Args:
            x (np.ndarray): Solution vector
            success (bool): Whether optimization was successful
            message (str): Optional message from optimizer
            n_iter (int): Number of iterations
            n_evals (int): Number of function evaluations
            time (float): Execution time in seconds
            
        Returns:
            dict: Standardized results dictionary
        """
        try:
            if success:
                x_arr = np.asarray(x)
                stage_ratios, mass_ratios = self.calculate_stage_ratios(x_arr)
                payload_fraction = self.calculate_fitness(x_arr)
                delta_v = self.calculate_delta_v(stage_ratios)
                
                return {
                    'success': True,
                    'x': x_arr.tolist(),
                    'fun': float(self.objective_with_penalty(x_arr)),
                    'payload_fraction': float(payload_fraction),
                    'stage_ratios': stage_ratios.tolist(),
                    'mass_ratios': mass_ratios.tolist(),
                    'stages': self.create_stage_results(x_arr, stage_ratios),
                    'dv': delta_v.tolist(),
                    'method': self.__class__.__name__,
                    'n_iterations': int(n_iter),
                    'n_function_evals': int(n_evals),
                    'execution_time': float(time)
                }
            else:
                return {
                    'success': False,
                    'message': str(message),
                    'method': self.__class__.__name__
                }
        except Exception as e:
            logger.error(f"Error processing results: {e}")
            return {
                'success': False,
                'message': f"Error processing results: {str(e)}",
                'method': self.__class__.__name__
            }
        
    @abstractmethod
    def solve(self, initial_guess, bounds):
        """Solve the optimization problem.
        
        Args:
            initial_guess: Initial solution guess
            bounds: List of (min, max) bounds for each variable
            
        Returns:
            dict: Optimization results
        """
        pass
