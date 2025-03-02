"""PyMOO problem definition for rocket stage optimization."""
import numpy as np
from pymoo.core.problem import Problem
from src.utils.config import logger
from src.optimization.objective import payload_fraction_objective, enforce_stage_constraints, calculate_payload_fraction
from src.optimization.physics import calculate_stage_ratios
from src.optimization.cache import OptimizationCache
from src.optimization.parallel_solver import ParallelSolver

class RocketOptimizationProblem(Problem):
    """Problem definition for rocket stage optimization."""
    
    def __init__(self, n_var, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config=None):
        """Initialize the optimization problem.
        
        Args:
            n_var: Number of variables (stages)
            bounds: List of (min, max) bounds for each variable
            G0: Gravitational constant
            ISP: List of specific impulse values
            EPSILON: List of structural fraction values
            TOTAL_DELTA_V: Total required delta-v
            config: Configuration dictionary
        """
        xl = np.array([b[0] for b in bounds])
        xu = np.array([b[1] for b in bounds])
        super().__init__(n_var=n_var, n_obj=1, n_constr=1, xl=xl, xu=xu)
        
        self.G0 = G0
        self.ISP = np.asarray(ISP, dtype=float)
        self.EPSILON = np.asarray(EPSILON, dtype=float)
        self.TOTAL_DELTA_V = TOTAL_DELTA_V
        self.config = config if config is not None else {}
        self.cache = OptimizationCache()
    
    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate solutions.
        
        Args:
            x: Solution or population of solutions
            out: Output dictionary for fitness and constraints
        """
        # Evaluate each solution
        f = np.zeros((x.shape[0], 1))
        g = np.zeros((x.shape[0], 1))
        
        for i in range(x.shape[0]):
            # Check cache first
            solution = x[i]
            cached_fitness = self.cache.get_cached_fitness(solution)
            
            if cached_fitness is not None:
                payload_fraction = cached_fitness
                logger.debug(f"Cache hit for solution {i}")
            else:
                # Calculate stage ratios and payload fraction
                stage_ratios, mass_ratios = calculate_stage_ratios(solution, self.G0, self.ISP, self.EPSILON)
                payload_fraction = calculate_payload_fraction(mass_ratios)
                # Store in cache
                self.cache.add(solution, payload_fraction)
                logger.debug(f"Cache miss for solution {i}")
            
            # Store objective (negative since we're minimizing)
            f[i, 0] = -payload_fraction
            
            # Calculate constraint violations with scaling
            violation = enforce_stage_constraints(solution, self.TOTAL_DELTA_V, self.config)
            # Scale violations to help solver navigate constraint space
            if violation > 0:
                if violation > 1.0:  # Major violation
                    g[i, 0] = 1000.0 * violation
                elif violation > 0.1:  # Moderate violation
                    g[i, 0] = 100.0 * violation
                else:  # Minor violation
                    g[i, 0] = 10.0 * violation
            else:
                g[i, 0] = 0.0
        
        out["F"] = f
        out["G"] = g

def get_solver_config(config, solver_name):
    """Get solver-specific configuration with defaults.
    
    Args:
        config: Main configuration dictionary
        solver_name: Name of the solver (e.g., 'ga', 'adaptive_ga', 'pso')
        
    Returns:
        dict: Solver configuration with defaults applied
    """
    # Get optimization section with defaults
    opt_config = config.get('optimization', {})
    
    # Get solver specific config from solvers section
    solver_config = opt_config.get('solvers', {}).get(solver_name, {})
    
    # Get constraints from main optimization config
    constraints = opt_config.get('constraints', {})
    
    # Common defaults
    defaults = {
        'penalty_coefficient': opt_config.get('penalty_coefficient', 1e3),
        'constraints': constraints,  # Include constraints from main config
        'solver_specific': {
            'population_size': 50,
            'n_generations': 100,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8
        }
    }
    
    # Solver-specific defaults
    solver_defaults = {
        'ga': {
            'solver_specific': {
                'population_size': 100,
                'n_generations': 200,
                'mutation': {
                    'eta': 20,
                    'prob': 0.1
                },
                'crossover': {
                    'eta': 15,
                    'prob': 0.8
                }
            }
        },
        'adaptive_ga': {
            'solver_specific': {
                'population_size': 100,
                'n_generations': 200,
                'initial_mutation_rate': 0.1,
                'initial_crossover_rate': 0.8,
                'min_mutation_rate': 0.01,
                'max_mutation_rate': 0.5,
                'min_crossover_rate': 0.5,
                'max_crossover_rate': 0.95,
                'adaptation_rate': 0.1
            }
        },
        'pso': {
            'solver_specific': {
                'n_particles': 50,
                'n_iterations': 100,
                'w': 0.7,  # Inertia weight
                'c1': 1.5,  # Cognitive parameter
                'c2': 1.5   # Social parameter
            }
        }
    }
    
    # Get solver-specific defaults
    solver_default = solver_defaults.get(solver_name, defaults)
    
    # Deep merge configs
    result = defaults.copy()
    result.update(solver_default)
    if solver_config:
        for key, value in solver_config.items():
            if isinstance(value, dict) and key in result:
                result[key].update(value)
            else:
                result[key] = value
    
    return result
