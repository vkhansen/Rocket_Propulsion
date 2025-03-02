"""Objective functions for rocket stage optimization."""
import numpy as np
from typing import Tuple, Dict, Union
from .physics import calculate_stage_ratios, calculate_payload_fraction
from .parallel_solver import ParallelSolver
from .solver_config import get_solver_config
from ..utils.config import logger

def payload_fraction_objective(dv: np.ndarray, G0: float, ISP: np.ndarray, EPSILON: np.ndarray) -> float:
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
        payload_fraction = calculate_payload_fraction(mass_ratios)
        return float(-payload_fraction)  # Negative for minimization
    except Exception as e:
        logger.error(f"Error in objective calculation: {str(e)}")
        return 1e6  # Large penalty for failed calculations

def enforce_stage_constraints(dv_array, total_dv_required, config=None):
    """Enforce stage constraints and return constraint violation value.
    
    Args:
        dv_array: Array of stage delta-v values
        total_dv_required: Required total delta-v
        config: Configuration dictionary containing constraints
        
    Returns:
        float: Constraint violation value (0 if all constraints satisfied)
    """
    if config is None:
        config = {}
    
    # Get constraint parameters from config
    constraints = config.get('constraints', {})
    total_dv_constraint = constraints.get('total_dv', {})
    tolerance = total_dv_constraint.get('tolerance', 1e-6)
    
    # Calculate total delta-v constraint violation
    total_dv = np.sum(dv_array)
    dv_violation = abs(total_dv - total_dv_required)
    
    # Get stage fraction constraints
    stage_fractions = constraints.get('stage_fractions', {})
    first_stage = stage_fractions.get('first_stage', {})
    other_stages = stage_fractions.get('other_stages', {})
    
    # Default constraints if not specified
    min_fraction_first = first_stage.get('min_fraction', 0.15)
    max_fraction_first = first_stage.get('max_fraction', 0.80)
    min_fraction_other = other_stages.get('min_fraction', 0.01)
    max_fraction_other = other_stages.get('max_fraction', 1.0)
    
    # Calculate stage fractions
    stage_fractions = dv_array / total_dv if total_dv > 0 else np.zeros_like(dv_array)
    
    total_violation = dv_violation
    
    # Check first stage constraints
    if len(stage_fractions) > 0:
        if stage_fractions[0] < min_fraction_first:
            total_violation += abs(stage_fractions[0] - min_fraction_first)
        if stage_fractions[0] > max_fraction_first:
            total_violation += abs(stage_fractions[0] - max_fraction_first)
    
    # Check other stage constraints
    for fraction in stage_fractions[1:]:
        if fraction < min_fraction_other:
            total_violation += abs(fraction - min_fraction_other)
        if fraction > max_fraction_other:
            total_violation += abs(fraction - max_fraction_other)
    
    return total_violation

def calculate_mass_ratios(stage_ratios, EPSILON):
    """Calculate mass ratios from stage ratios.
    
    Args:
        stage_ratios (np.ndarray): Stage ratios (λ = mf/m0) for each stage
        EPSILON (np.ndarray): Structural coefficients for each stage
        
    Returns:
        np.ndarray: Mass ratios (μ) for each stage
    """
    try:
        # Convert inputs to numpy arrays
        stage_ratios = np.asarray(stage_ratios, dtype=float)
        EPSILON = np.asarray(EPSILON, dtype=float)
        
        # Calculate mass ratios using corrected formula for mf/m0
        mass_ratios = np.zeros_like(stage_ratios)
        for i in range(len(stage_ratios)):
            mass_ratios[i] = 1.0 / (stage_ratios[i] * (1.0 - EPSILON[i]) + EPSILON[i])
            
        return mass_ratios
        
    except Exception as e:
        logger.error(f"Error calculating mass ratios: {str(e)}")
        return np.ones_like(stage_ratios)

def objective_with_penalty(dv, G0, ISP, EPSILON, TOTAL_DELTA_V, return_tuple=False) -> Union[float, Tuple[float, float, float]]:
    """Calculate objective value with penalties for constraint violations.
    
    Args:
        dv: Array of delta-v values for each stage
        G0: Gravitational constant
        ISP: Array of specific impulse values for each stage
        EPSILON: Array of structural coefficients for each stage
        TOTAL_DELTA_V: Required total delta-v
        return_tuple: If True, returns (objective, dv_constraint, physical_constraint)
                     If False, returns penalized scalar objective
        
    Returns:
        Union[float, Tuple[float, float, float]]: Either a scalar objective value or
            a tuple of (objective, dv_constraint, physical_constraint)
    """
    try:
        # Convert inputs to numpy arrays
        dv = np.asarray(dv, dtype=float).reshape(-1)  # Ensure 1D array
        ISP = np.asarray(ISP, dtype=float)
        EPSILON = np.asarray(EPSILON, dtype=float)
        
        # Calculate stage ratios and mass ratios
        stage_ratios, mass_ratios = calculate_stage_ratios(dv, G0, ISP, EPSILON)
        
        # Calculate payload fraction
        payload_fraction = calculate_payload_fraction(mass_ratios)
        objective = -payload_fraction  # Negative for minimization
        
        # Calculate constraint violations with relative scaling
        total_dv = np.sum(dv)
        dv_constraint = abs(total_dv - TOTAL_DELTA_V) / TOTAL_DELTA_V  # Relative error
        
        # Physical constraints on stage ratios (should be between 0 and 1)
        # Scale violations by the magnitude of violation
        stage_ratio_min_violations = np.maximum(0, -stage_ratios)  # Violations below 0
        stage_ratio_max_violations = np.maximum(0, stage_ratios - 1)  # Violations above 1
        
        # Normalize physical constraints by number of stages
        physical_constraint = (np.sum(stage_ratio_min_violations) + np.sum(stage_ratio_max_violations)) / len(stage_ratios)
        
        if return_tuple:
            return (objective, dv_constraint, physical_constraint)
        else:
            # Return penalized scalar objective with adaptive penalties
            # Use smaller penalties for small violations to help solvers navigate the space
            dv_penalty = 100.0 if dv_constraint > 0.1 else 10.0
            physical_penalty = 100.0 if physical_constraint > 0.1 else 10.0
            
            return objective + dv_penalty * dv_constraint + physical_penalty * physical_constraint
        
    except Exception as e:
        logger.error(f"Error in objective calculation: {str(e)}")
        if return_tuple:
            return (float('inf'), float('inf'), float('inf'))
        else:
            return float('inf')

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
        
        # Get common parameters
        G0 = float(self.parameters.get('G0', 9.81))
        TOTAL_DELTA_V = float(self.parameters.get('TOTAL_DELTA_V', 0.0))
        ISP = [float(stage['ISP']) for stage in self.stages]
        EPSILON = [float(stage['EPSILON']) for stage in self.stages]
        bounds = [(0, TOTAL_DELTA_V) for _ in range(len(self.stages))]
        
        # Get solver configurations
        slsqp_config = get_solver_config(self.config, 'slsqp')
        ga_config = get_solver_config(self.config, 'ga')
        adaptive_ga_config = get_solver_config(self.config, 'adaptive_ga')
        pso_config = get_solver_config(self.config, 'pso')
        de_config = get_solver_config(self.config, 'de')
        basin_config = get_solver_config(self.config, 'basin')
        
        # Create solver instances with configs
        return [
            SLSQPSolver(G0=G0, ISP=ISP, EPSILON=EPSILON, TOTAL_DELTA_V=TOTAL_DELTA_V, bounds=bounds, config=slsqp_config),
            GeneticAlgorithmSolver(G0=G0, ISP=ISP, EPSILON=EPSILON, TOTAL_DELTA_V=TOTAL_DELTA_V, bounds=bounds, config=ga_config),
            AdaptiveGeneticAlgorithmSolver(G0=G0, ISP=ISP, EPSILON=EPSILON, TOTAL_DELTA_V=TOTAL_DELTA_V, bounds=bounds, config=adaptive_ga_config),
            ParticleSwarmOptimizer(G0=G0, ISP=ISP, EPSILON=EPSILON, TOTAL_DELTA_V=TOTAL_DELTA_V, bounds=bounds, config=pso_config),
            DifferentialEvolutionSolver(G0=G0, ISP=ISP, EPSILON=EPSILON, TOTAL_DELTA_V=TOTAL_DELTA_V, bounds=bounds, config=de_config),
            BasinHoppingOptimizer(G0=G0, ISP=ISP, EPSILON=EPSILON, TOTAL_DELTA_V=TOTAL_DELTA_V, bounds=bounds, config=basin_config)
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
