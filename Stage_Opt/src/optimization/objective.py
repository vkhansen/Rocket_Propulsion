"""Objective functions for rocket stage optimization."""
import numpy as np
from typing import Tuple, Dict
from .physics import calculate_stage_ratios, calculate_payload_fraction
from ..utils.config import logger
from .parallel_solver import ParallelSolver

def calculate_objective(dv: np.ndarray, G0: float, ISP: np.ndarray, EPSILON: np.ndarray) -> float:
    """Calculate the objective value (negative payload fraction).
    
    Args:
        dv: Array of delta-v values for each stage
        G0: Gravitational constant
        ISP: Array of specific impulse values for each stage
        EPSILON: Array of structural coefficients for each stage
        
    Returns:
        float: Negative payload fraction (for minimization)
    """
    try:
        stage_ratios, mass_ratios = calculate_stage_ratios(dv, G0, ISP, EPSILON)
        payload_fraction = calculate_payload_fraction(mass_ratios, EPSILON)
        return float(-payload_fraction)  # Negative for minimization
    except Exception as e:
        logger.error(f"Error in objective calculation: {str(e)}")
        return 1e6  # Large penalty for failed calculations

def objective_with_penalty(dv, G0, ISP, EPSILON, TOTAL_DELTA_V) -> Tuple[float, float, float]:
    """Calculate objective value with penalties for constraint violations.
    
    Args:
        dv: Array of delta-v values for each stage
        G0: Gravitational constant
        ISP: Array of specific impulse values for each stage
        EPSILON: Array of structural coefficients for each stage
        TOTAL_DELTA_V: Required total delta-v
        
    Returns:
        Tuple[float, float, float]: (objective, dv_constraint, physical_constraint)
            - objective: Negative payload fraction (for minimization)
            - dv_constraint: Delta-v constraint violation
            - physical_constraint: Physical constraint violation
    """
    try:
        # Convert inputs to numpy arrays
        dv = np.asarray(dv, dtype=float).reshape(-1)  # Ensure 1D array
        ISP = np.asarray(ISP, dtype=float)
        EPSILON = np.asarray(EPSILON, dtype=float)
        
        # Calculate stage ratios and mass ratios
        stage_ratios, mass_ratios = calculate_stage_ratios(dv, G0, ISP, EPSILON)
        
        # Calculate payload fraction
        payload_fraction = calculate_payload_fraction(mass_ratios, EPSILON)
        
        # Calculate constraint violations
        dv_constraint = abs(np.sum(dv) - TOTAL_DELTA_V)
        
        # Physical constraints on stage ratios (should be between 0 and 1)
        physical_constraint = np.sum(np.maximum(0, -stage_ratios)) + np.sum(np.maximum(0, stage_ratios - 1))
        
        # Return objective and constraints separately
        return (-payload_fraction, dv_constraint, physical_constraint)
        
    except Exception as e:
        logger.error(f"Error in objective calculation: {str(e)}")
        return (float('inf'), float('inf'), float('inf'))  # Return tuple of infinities for failed calculations

def get_constraint_violations(dv: np.ndarray, G0: float, ISP: np.ndarray, 
                            EPSILON: np.ndarray, TOTAL_DELTA_V: float) -> Tuple[float, float]:
    """Calculate constraint violations.
    
    Args:
        dv: Array of delta-v values for each stage
        G0: Gravitational constant
        ISP: Array of specific impulse values for each stage
        EPSILON: Array of structural coefficients for each stage
        TOTAL_DELTA_V: Required total delta-v
        
    Returns:
        tuple: (dv_constraint, physical_constraint)
            - dv_constraint: Delta-v constraint violation
            - physical_constraint: Physical constraint violation
    """
    try:
        dv = np.asarray(dv, dtype=float).reshape(-1)
        stage_ratios, _ = calculate_stage_ratios(dv, G0, ISP, EPSILON)
        
        # Delta-v constraint
        dv_constraint = float(abs(np.sum(dv) - TOTAL_DELTA_V))
        
        # Physical constraints
        physical_constraint = float(np.sum(np.maximum(0, -stage_ratios)) + 
                                 np.sum(np.maximum(0, stage_ratios - 1)))
        
        return dv_constraint, physical_constraint
        
    except Exception as e:
        logger.error(f"Error calculating constraints: {str(e)}")
        return float('inf'), float('inf')

class RocketStageOptimizer:
    """Class to manage rocket stage optimization using different solvers."""
    
    def __init__(self, config, parameters, stages):
        """Initialize the optimizer with configuration and parameters."""
        self.config = config
        self.parameters = parameters
        self.stages = stages
        self.solvers = []  # Initialize solvers after imports
        
    def _initialize_solvers(self):
        """Initialize all available solvers."""
        # Import solvers here to avoid circular imports
        from .solvers.slsqp_solver import SLSQPSolver
        from .solvers.ga_solver import GeneticAlgorithmSolver
        from .solvers.adaptive_ga_solver import AdaptiveGeneticAlgorithmSolver
        from .solvers.pso_solver import ParticleSwarmOptimizer
        from .solvers.de_solver import DifferentialEvolutionSolver
        from .solvers.basin_hopping_solver import BasinHoppingOptimizer
        
        # Create problem parameters dictionary
        problem_params = {
            'G0': float(self.parameters.get('G0', 9.81)),
            'TOTAL_DELTA_V': float(self.parameters.get('TOTAL_DELTA_V', 0.0)),
            'stages': self.stages
        }
        
        return [
            SLSQPSolver(self.config, problem_params),
            GeneticAlgorithmSolver(self.config, problem_params),
            AdaptiveGeneticAlgorithmSolver(self.config, problem_params),
            ParticleSwarmOptimizer(self.config, problem_params),
            DifferentialEvolutionSolver(self.config, problem_params),
            BasinHoppingOptimizer(self.config, problem_params)
        ]
    
    def solve(self, initial_guess, bounds):
        """Run optimization with all available solvers in parallel."""
        if not self.solvers:
            self.solvers = self._initialize_solvers()
        
        # Configure parallel solver
        parallel_config = self.config.get('parallel', {})
        if not parallel_config:
            parallel_config = {
                'max_workers': None,  # Use all available CPUs
                'timeout': 3600,      # 1 hour total timeout
                'solver_timeout': 600  # 10 minutes per solver
            }
        
        # Initialize parallel solver
        parallel_solver = ParallelSolver(parallel_config)
        
        try:
            # Run all solvers in parallel and return results directly
            # The parallel solver now returns results in the format expected by reporting
            results = parallel_solver.solve(self.solvers, initial_guess, bounds)
            
            if not results:
                logger.warning("No solutions found from any solver")
                return {}
                
            logger.info(f"Successfully completed parallel optimization with {len(results)} solutions")
            return results
            
        except Exception as e:
            logger.error(f"Error in parallel optimization: {str(e)}")
            return {}
