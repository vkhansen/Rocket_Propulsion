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
    min_fraction_other = other_stages_cfg.get('min_fraction', 0.01)
    max_fraction_other = other_stages_cfg.get('max_fraction', 1.0)

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


def objective_with_penalty(dv: np.ndarray,
                           G0: float,
                           ISP: np.ndarray,
                           EPSILON: np.ndarray,
                           TOTAL_DELTA_V: float,
                           config: Dict = None,
                           return_tuple: bool = False) -> Union[float, Tuple[float, float, float]]:
    """Main objective function with continuous penalties for constraints."""

    if config is None:
        config = {}

    try:
        # Ensure arrays
        dv = np.asarray(dv, dtype=float).flatten()
        ISP = np.asarray(ISP, dtype=float)
        EPSILON = np.asarray(EPSILON, dtype=float)

        # Calculate the negative payload fraction
        stage_ratios, mass_ratios = calculate_stage_ratios(dv, G0, ISP, EPSILON)
        payload_fraction = calculate_payload_fraction(mass_ratios)

        # Reject if payload fraction is nonphysical
        if payload_fraction <= 0:
            if return_tuple:
                return (float('inf'), float('inf'), float('inf'))
            return float('inf')

        objective_value = -payload_fraction  # We minimize the negative => maximize fraction

        # 1) Basic physical constraint: stage_ratios must be between 0 and 1
        #    We'll treat any violation as a separate measure. 
        below_zero_viol = np.sum(np.maximum(0.0, -stage_ratios))      # how far below 0 
        above_one_viol  = np.sum(np.maximum(0.0, stage_ratios - 1.0)) # how far above 1
        physical_constraint = (below_zero_viol + above_one_viol) / len(stage_ratios)

        # 2) Delta-v constraint as a relative error from required total
        total_dv = np.sum(dv)
        dv_constraint = abs(total_dv - TOTAL_DELTA_V) / max(1e-8, TOTAL_DELTA_V)

        # 3) Stage fraction constraints via new function
        #    (this checks min/max fraction for 1st stage vs others, etc.)
        fraction_violation = enforce_stage_constraints(dv,
                                                       total_dv_required=TOTAL_DELTA_V,
                                                       config=config)

        # Combine all constraints into a single penalty. 
        # You can tune these scales so the solver doesn't get stuck:
        # For example, bigger penalty_scale => solver tries harder to fix constraint violations.
        penalty_scale_phys = 10
        penalty_scale_dv   = 10
        penalty_scale_frac = 10

        # Weighted sum of constraints
        # More sophisticated approaches might weight them differently, 
        # or do a piecewise approach. 
        total_penalty = penalty_scale_phys * physical_constraint \
                        + penalty_scale_dv   * dv_constraint       \
                        + penalty_scale_frac * fraction_violation

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
