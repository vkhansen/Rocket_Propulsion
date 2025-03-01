"""Objective functions for optimization."""
import numpy as np
from ..utils.config import logger
from .physics import calculate_mass_ratios, calculate_payload_fraction, calculate_stage_ratios
from .parallel_solver import ParallelSolver

def payload_fraction_objective(dv, G0, ISP, EPSILON):
    """Calculate the payload fraction objective using the corrected physics model."""
    try:
        logger.debug(f"Evaluating payload fraction objective with dv={dv}")
        
        # Calculate mass ratios and payload fraction
        stage_ratios, mass_ratios = calculate_stage_ratios(dv, G0, ISP, EPSILON)
        payload_fraction = calculate_payload_fraction(mass_ratios, EPSILON)
        
        return -payload_fraction  # Negative for minimization
        
    except Exception as e:
        logger.error(f"Error in payload fraction objective: {e}")
        return 1e6  # Large penalty for failed calculations

def objective_with_penalty(dv, G0, ISP, EPSILON, TOTAL_DELTA_V):
    """Calculate objective value with penalty for constraint violations.
    
    Args:
        dv (np.ndarray): Delta-v values for each stage
        G0 (float): Gravitational acceleration
        ISP (np.ndarray): Specific impulse for each stage
        EPSILON (np.ndarray): Structural coefficients for each stage
        TOTAL_DELTA_V (float): Total required delta-v
        
    Returns:
        float: Objective value (negative payload fraction) with penalties
    """
    try:
        # Convert inputs to numpy arrays
        dv = np.asarray(dv, dtype=float)
        ISP = np.asarray(ISP, dtype=float)
        EPSILON = np.asarray(EPSILON, dtype=float)
        
        # Calculate stage ratios and mass ratios using physics module
        stage_ratios, mass_ratios = calculate_stage_ratios(dv, G0, ISP, EPSILON)
        
        # Calculate payload fraction
        payload_fraction = 1.0
        for ratio in mass_ratios:
            payload_fraction /= ratio
        
        # Calculate constraint violations
        total_dv = float(np.sum(dv))
        dv_violation = abs(total_dv - TOTAL_DELTA_V)
        negative_violation = np.sum(np.abs(np.minimum(dv, 0)))
        
        # Apply much stronger penalties
        penalty_coefficient = 1e6  # Increased from 1e3 to 1e6
        penalty = dv_violation + negative_violation
        
        # Add physical constraint penalties
        physical_penalty = 0.0
        min_stage_ratio = 0.1  # Minimum allowable stage ratio
        max_stage_ratio = 0.9  # Maximum allowable stage ratio
        
        # Penalize stage ratios outside physical bounds
        for ratio in stage_ratios:
            if ratio < min_stage_ratio or ratio > max_stage_ratio:
                physical_penalty += 1.0
                
        # Add strong penalty for unrealistic stage ratios
        penalty += physical_penalty * penalty_coefficient
        
        result = float(payload_fraction) - penalty_coefficient * penalty
        
        return -result  # Negative because we want to maximize payload fraction
        
    except Exception as e:
        logger.error(f"Error calculating objective: {str(e)}")
        return float('inf')

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
            # Run all solvers in parallel
            results = parallel_solver.solve(self.solvers, initial_guess, bounds)
            
            if not results:
                logger.warning("No solutions found from any solver")
                return {}
                
            logger.info(f"Successfully completed parallel optimization with {len(results)} solutions")
            return results
            
        except Exception as e:
            logger.error(f"Error in parallel optimization: {str(e)}")
            return {}
