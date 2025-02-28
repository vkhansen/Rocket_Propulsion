"""Objective functions for optimization."""
import numpy as np
from ..utils.data import calculate_mass_ratios, calculate_payload_fraction
from ..utils.config import logger
from .solvers.slsqp_solver import SLSQPSolver
from .solvers.ga_solver import GeneticAlgorithmSolver
from .solvers.adaptive_ga_solver import AdaptiveGeneticAlgorithmSolver
from .solvers.pso_solver import ParticleSwarmOptimizer
from .solvers.de_solver import DifferentialEvolutionSolver
from .solvers.basin_hopping_solver import BasinHoppingOptimizer

def payload_fraction_objective(dv, G0, ISP, EPSILON):
    """Calculate the payload fraction objective using the corrected physics model."""
    try:
        logger.debug(f"Evaluating payload fraction objective with dv={dv}")
        
        # Ensure all inputs are numpy arrays
        dv = np.asarray(dv, dtype=float)
        ISP = np.asarray(ISP, dtype=float)
        EPSILON = np.asarray(EPSILON, dtype=float)
        
        # Pass G0 to calculate_mass_ratios so that the negative exponent is used
        mass_ratios = calculate_mass_ratios(dv, ISP, EPSILON, G0)
        payload_fraction = calculate_payload_fraction(mass_ratios)
        
        # Add a small penalty for solutions close to constraint violations
        penalty = 0.0
        for ratio in mass_ratios:
            if ratio <= 0.1:  # Penalize solutions close to physical limits
                penalty += 100.0 * (0.1 - ratio) ** 2
                
        # Negative for minimization
        result = float(-payload_fraction + penalty)
        logger.debug(f"Payload fraction objective: {result} (payload={payload_fraction}, penalty={penalty})")
        return result
        
    except Exception as e:
        logger.error(f"Error in payload fraction calculation: {e}")
        return 1e6  # Large but finite penalty

def calculate_stage_ratios(dv, G0, ISP, EPSILON):
    """Calculate stage ratios (λ) for each stage.
    
    Args:
        dv: Stage delta-v values
        G0: Gravitational constant
        ISP: Specific impulse values
        EPSILON: Structural fraction values
        
    Returns:
        tuple: (stage_ratios, mass_ratios)
            - stage_ratios: List of stage ratios (λ)
            - mass_ratios: List of mass ratios before structural fraction
    """
    stage_ratios = []
    mass_ratios = []
    for dv_i, isp, eps in zip(dv, ISP, EPSILON):
        mass_ratio = np.exp(-dv_i / (G0 * isp))
        lambda_val = mass_ratio * (1 - eps)  # Corrected λᵢ calculation
        stage_ratios.append(lambda_val)
        mass_ratios.append(mass_ratio)
    return stage_ratios, mass_ratios

def objective_with_penalty(dv, G0, ISP, EPSILON, TOTAL_DELTA_V):
    """Calculate objective with penalty for constraint violation."""
    try:
        logger.debug(f"Evaluating objective with penalty: dv={dv}")
        
        # Ensure all inputs are numpy arrays
        dv = np.asarray(dv, dtype=float)
        ISP = np.asarray(ISP, dtype=float)
        EPSILON = np.asarray(EPSILON, dtype=float)
        
        # Base objective
        base_obj = payload_fraction_objective(dv, G0, ISP, EPSILON)
        
        # Constraint violation penalty
        dv_sum = float(np.sum(dv))
        constraint_violation = abs(dv_sum - TOTAL_DELTA_V)
        penalty = 1e3 * constraint_violation  # Reduced penalty coefficient
        
        result = float(base_obj + penalty)
        logger.debug(f"Objective with penalty: {result} (base={base_obj}, penalty={penalty})")
        return result
        
    except Exception as e:
        logger.error(f"Error in objective calculation: {e}")
        return 1e6  # Large but finite penalty

class RocketStageOptimizer:
    """Class to manage rocket stage optimization using different solvers."""
    
    def __init__(self, config, parameters, stages):
        """Initialize the optimizer with configuration and parameters."""
        self.config = config
        self.parameters = parameters
        self.stages = stages
        self.solvers = self._initialize_solvers()
        
    def _initialize_solvers(self):
        """Initialize all available solvers."""
        return [
            SLSQPSolver(self.config, self.parameters, self.stages),
            GeneticAlgorithmSolver(self.config, self.parameters, self.stages),
            AdaptiveGeneticAlgorithmSolver(self.config, self.parameters, self.stages),
            ParticleSwarmOptimizer(self.config, self.parameters, self.stages),
            DifferentialEvolutionSolver(self.config, self.parameters, self.stages),
            BasinHoppingOptimizer(self.config, self.parameters, self.stages)
        ]
    
    def solve(self, initial_guess, bounds):
        """Run optimization with all available solvers."""
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
