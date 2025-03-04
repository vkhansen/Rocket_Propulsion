import numpy as np
from typing import Tuple, Dict, Union
from .physics import calculate_stage_ratios, calculate_payload_fraction
from .parallel_solver import ParallelSolver
from .solver_config import get_solver_config
from ..utils.config import logger

def enforce_stage_constraints(dv_array: np.ndarray,
                              total_dv_required: float,
                              config: Dict = None) -> float:
    """Enforce stage constraints and return a total violation penalty.

    This returns a *continuous* violation measure (>= 0). 
    A result of 0 means no violations.
    A larger positive number means larger constraint violations.
    """
    if config is None:
        config = {}
    
    constraints = config.get('constraints', {})
    total_dv_constraint = constraints.get('total_dv', {})
    tolerance = total_dv_constraint.get('tolerance', 1e-6)

    # Calculate total Delta-V
    total_dv = np.sum(dv_array)
    # DEBUG prints/logs
    logger.debug(f"dv_array={dv_array}, sum={np.sum(dv_array)}")
    
    # If within tolerance, treat it as zero violation
    dv_violation_raw = abs(total_dv - total_dv_required)
    dv_violation = 0.0 if dv_violation_raw <= tolerance else dv_violation_raw

    # Stage‐fraction constraints
    stage_fractions_cfg = constraints.get('stage_fractions', {})
    first_stage_cfg = stage_fractions_cfg.get('first_stage', {})
    other_stages_cfg = stage_fractions_cfg.get('other_stages', {})

    # Fallback defaults if not in config
    min_fraction_first = first_stage_cfg.get('min_fraction', 0.15)
    max_fraction_first = first_stage_cfg.get('max_fraction', 0.80)
    min_fraction_other = other_stages_cfg.get('min_fraction', 0.1)
    max_fraction_other = other_stages_cfg.get('max_fraction', 0.6)

    # Compute fractions if total_dv > 0
    if total_dv > 0:
        stage_fractions = dv_array / total_dv
    else:
        stage_fractions = np.zeros_like(dv_array)
    
    # DEBUG prints/logs
    logger.debug(f"fractions={stage_fractions}")

    total_violation = dv_violation

    # Check first stage
    if len(stage_fractions) > 0:
        if stage_fractions[0] < min_fraction_first:
            total_violation += abs(stage_fractions[0] - min_fraction_first)
        if stage_fractions[0] > max_fraction_first:
            total_violation += abs(stage_fractions[0] - max_fraction_first)

    # Check other stages
    for frac in stage_fractions[1:]:
        if frac < min_fraction_other:
            total_violation += abs(frac - min_fraction_other)
        if frac > max_fraction_other:
            total_violation += abs(frac - max_fraction_other)

    return total_violation


def payload_fraction_objective(dv: np.ndarray,
                               G0: float,
                               ISP: np.ndarray,
                               EPSILON: np.ndarray) -> float:
    """Calculate the objective value (negative payload fraction)."""
    try:
        stage_ratios, mass_ratios = calculate_stage_ratios(dv, G0, ISP, EPSILON)
        payload_fraction = calculate_payload_fraction(mass_ratios)
        if payload_fraction <= 0:
            # Hard reject
            logger.warning(f"Rejecting solution with nonphysical payload fraction: {payload_fraction}")
            return float('inf')
        # Negative of payload fraction (we want to *maximize* fraction => minimize negative)
        return -payload_fraction
    except Exception as e:
        logger.error(f"Error in objective calculation: {str(e)}")
        return float('inf')


def objective_with_penalty(dv, G0, ISP, EPSILON, TOTAL_DELTA_V, return_tuple=False):
    """Calculate objective function with penalty for constraint violations.
    
    Args:
        dv: Delta-V distribution for each stage
        G0: Gravitational constant
        ISP: Specific impulse for each stage
        EPSILON: Structural coefficient for each stage
        TOTAL_DELTA_V: Total required delta-V
        return_tuple: If True, return (objective, dv_constraint, physical_constraint)
        
    Returns:
        Penalized objective value, or tuple if return_tuple=True
    """
    try:
        # Convert to numpy array if not already
        dv = np.asarray(dv, dtype=np.float64)
        ISP = np.asarray(ISP, dtype=np.float64)
        EPSILON = np.asarray(EPSILON, dtype=np.float64)
        
        # Check for NaN or inf values
        if not np.all(np.isfinite(dv)) or not np.all(np.isfinite(ISP)) or not np.all(np.isfinite(EPSILON)):
            logger.error(f"Non-finite values detected: dv={dv}, ISP={ISP}, EPSILON={EPSILON}")
            if return_tuple:
                return (float('inf'), float('inf'), float('inf'))
            return float('inf')
        
        # Calculate total delta-v constraint violation
        total_dv = np.sum(dv)
        dv_constraint = abs(total_dv - TOTAL_DELTA_V) / TOTAL_DELTA_V
        
        # Calculate stage fractions for logging
        stage_fractions = dv / TOTAL_DELTA_V if TOTAL_DELTA_V > 0 else np.zeros_like(dv)
        
        # Check for negative values which would cause physics issues
        if np.any(dv <= 0) or np.any(ISP <= 0) or np.any(EPSILON <= 0) or np.any(EPSILON >= 1):
            logger.warning(f"Invalid physics parameters: dv={dv}, ISP={ISP}, EPSILON={EPSILON}")
            if return_tuple:
                return (float('inf'), dv_constraint, float('inf'))
            return float('inf')
        
        # Calculate mass ratios for each stage
        try:
            exp_terms = np.exp(dv / (ISP * G0))
            
            # Check for numerical overflow
            if np.any(~np.isfinite(exp_terms)):
                logger.warning(f"Numerical overflow in exp_terms: {exp_terms}")
                if return_tuple:
                    return (float('inf'), dv_constraint, float('inf'))
                return float('inf')
                
            # Calculate lambda (stage ratio) and mu (mass ratio)
            lambda_values = 1.0 / exp_terms
            mu_values = 1.0 / (lambda_values * (1.0 - EPSILON) + EPSILON)
            
            # Check for numerical issues
            if np.any(~np.isfinite(lambda_values)) or np.any(~np.isfinite(mu_values)):
                logger.warning(f"Numerical issues: lambda={lambda_values}, mu={mu_values}")
                if return_tuple:
                    return (float('inf'), dv_constraint, float('inf'))
                return float('inf')
                
            # Calculate payload fraction (product of 1/mu values)
            payload_fraction = np.prod(1.0 / mu_values)
            
            # Check for numerical issues
            if not np.isfinite(payload_fraction) or payload_fraction <= 0:
                logger.warning(f"Invalid payload fraction: {payload_fraction}")
                if return_tuple:
                    return (float('inf'), dv_constraint, float('inf'))
                return float('inf')
                
            # Objective is negative payload fraction (for minimization)
            objective_value = -payload_fraction
            
        except Exception as e:
            logger.error(f"Physics calculation error: {str(e)}")
            if return_tuple:
                return (float('inf'), dv_constraint, float('inf'))
            return float('inf')
        
        # Physical constraint violation
        physical_constraint = 0.0
        
        # Check for invalid stage ratios
        if np.any(lambda_values <= 0) or np.any(lambda_values >= 1):
            physical_constraint += 1e3
        
        # Check for invalid mass ratios
        if np.any(mu_values <= 1):
            physical_constraint += 1e3
        
        # Stage fraction violations
        fraction_violation = 0.0
        
        # Log the solution details
        logger.debug(f"DV: {dv}, Total: {total_dv}, Target: {TOTAL_DELTA_V}")
        logger.debug(f"Stage fractions: {stage_fractions}")
        logger.debug(f"Lambda: {lambda_values}, Mu: {mu_values}")
        logger.debug(f"Payload fraction: {payload_fraction}")
        logger.debug(f"DV constraint: {dv_constraint}, Physical constraint: {physical_constraint}")
        
        # Penalty scaling factors
        penalty_scale_phys = 1e3
        penalty_scale_dv = 1e3
        penalty_scale_frac = 5.0  # Reduced from 10

        # Weighted sum of constraints
        total_penalty = penalty_scale_phys * physical_constraint \
                        + penalty_scale_dv   * dv_constraint       \
                        + penalty_scale_frac * fraction_violation

        # Log the penalty and final objective
        logger.debug(f"Total penalty: {total_penalty}, Objective value: {objective_value}")

        # If the penalty is very large, cap it to avoid numerical issues
        if total_penalty > 1e6:
            logger.debug(f"Capping very large penalty: {total_penalty} to 1e6")
            total_penalty = 1e6

        penalized_objective = objective_value + total_penalty

        if return_tuple:
            return (penalized_objective, dv_constraint, physical_constraint)
        else:
            return penalized_objective

    except Exception as e:
        logger.error(f"Error in objective calculation: {str(e)}")
        if return_tuple:
            return (float('inf'), float('inf'), float('inf'))
        return float('inf')


class RocketStageOptimizer:
    """Class to manage rocket stage optimization using different solvers."""
    
    def __init__(self, config, parameters, stages):
        self.config = config
        self.parameters = parameters
        self.stages = stages
        self.solvers = []
        
    def _initialize_solvers(self):
        from .solvers.slsqp_solver import SLSQPSolver
        from .solvers.ga_solver import GeneticAlgorithmSolver
        from .solvers.adaptive_ga_solver import AdaptiveGeneticAlgorithmSolver
        from .solvers.pso_solver import ParticleSwarmOptimizer
        from .solvers.de_solver import DifferentialEvolutionSolver
        from .solvers.basin_hopping_solver import BasinHoppingOptimizer
        
        G0 = float(self.parameters.get('G0', 9.81))
        TOTAL_DELTA_V = float(self.parameters.get('TOTAL_DELTA_V', 0.0))
        ISP = [float(stage['ISP']) for stage in self.stages]
        EPSILON = [float(stage['EPSILON']) for stage in self.stages]
        n_stages = len(self.stages)
        bounds = [(0, TOTAL_DELTA_V) for _ in range(n_stages)]
        
        # Load solver configs
        slsqp_config  = get_solver_config(self.config, 'slsqp')
        ga_config     = get_solver_config(self.config, 'ga')
        adaptive_ga_config = get_solver_config(self.config, 'adaptive_ga')
        pso_config    = get_solver_config(self.config, 'pso')
        de_config     = get_solver_config(self.config, 'de')
        basin_config  = get_solver_config(self.config, 'basin')
        
        return [
            SLSQPSolver(G0=G0, ISP=ISP, EPSILON=EPSILON, TOTAL_DELTA_V=TOTAL_DELTA_V,
                        bounds=bounds, config=slsqp_config),
            GeneticAlgorithmSolver(G0=G0, ISP=ISP, EPSILON=EPSILON, TOTAL_DELTA_V=TOTAL_DELTA_V,
                                   bounds=bounds, config=ga_config),
            AdaptiveGeneticAlgorithmSolver(G0=G0, ISP=ISP, EPSILON=EPSILON, TOTAL_DELTA_V=TOTAL_DELTA_V,
                                           bounds=bounds, config=adaptive_ga_config),
            ParticleSwarmOptimizer(G0=G0, ISP=ISP, EPSILON=EPSILON, TOTAL_DELTA_V=TOTAL_DELTA_V,
                                   bounds=bounds, config=pso_config),
            DifferentialEvolutionSolver(G0=G0, ISP=ISP, EPSILON=EPSILON, TOTAL_DELTA_V=TOTAL_DELTA_V,
                                        bounds=bounds, config=de_config),
            BasinHoppingOptimizer(G0=G0, ISP=ISP, EPSILON=EPSILON, TOTAL_DELTA_V=TOTAL_DELTA_V,
                                  bounds=bounds, config=basin_config)
        ]
    
    def solve(self, initial_guess, bounds):
        if not self.solvers:
            self.solvers = self._initialize_solvers()
        
        # Parallel solver config
        parallel_config = self.config.get('parallel', {
            'max_workers': None,
            'timeout': 3600,
            'solver_timeout': 600
        })
        
        parallel_solver = ParallelSolver(parallel_config)
        
        try:
            results = parallel_solver.solve(self.solvers, initial_guess, bounds)
            if not results:
                logger.warning("No solutions found from any solver")
                return {}
            
            logger.info(f"Successfully completed optimization with {len(results)} solutions")
            return results

        except Exception as e:
            logger.error(f"Error in parallel optimization: {str(e)}")
            return {}
