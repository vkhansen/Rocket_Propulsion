"""Objective functions for optimization."""
import numpy as np
from ..utils.config import logger
from .physics import calculate_mass_ratios, calculate_payload_fraction, calculate_stage_ratios

def payload_fraction_objective(dv, G0, ISP, EPSILON):
    """Calculate the payload fraction objective using the corrected physics model."""
    try:
        logger.debug(f"Evaluating payload fraction objective with dv={dv}")
        
        # Calculate mass ratios and payload fraction
        mass_ratios = calculate_mass_ratios(dv, ISP, EPSILON, G0)
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
        
        # Calculate stage ratios
        stage_ratios = np.exp(dv / (G0 * ISP))
        
        # Calculate mass ratios
        mass_ratios = np.zeros_like(stage_ratios)
        for i in range(len(stage_ratios)):
            mass_ratios[i] = stage_ratios[i] / (1.0 - EPSILON[i] * (1.0 - stage_ratios[i]))
        
        # Calculate payload fraction
        payload_fraction = 1.0
        for ratio in mass_ratios:
            payload_fraction /= ratio
        
        # Calculate constraint violations
        total_dv = float(np.sum(dv))
        dv_violation = abs(total_dv - TOTAL_DELTA_V)
        negative_violation = np.sum(np.abs(np.minimum(dv, 0)))
        
        # Apply penalties
        penalty_coefficient = 1e3
        penalty = dv_violation + negative_violation
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
        """Run optimization with all available solvers."""
        if not self.solvers:
            self.solvers = self._initialize_solvers()
            
        results = {}
        
        for solver in self.solvers:
            try:
                logger.info(f"Starting {solver.name} optimization...")
                solution = solver.solve(initial_guess, bounds)
                
                if solution:
                    results[solver.name] = solution
                    logger.info(f"Successfully completed {solver.name} optimization")
                else:
                    logger.warning(f"{solver.name} optimization returned no solution")
                    
            except Exception as e:
                logger.error(f"Error in {solver.name} optimization: {str(e)}")
                results[solver.name] = {
                    'success': False,
                    'message': str(e)
                }
        
        return results
