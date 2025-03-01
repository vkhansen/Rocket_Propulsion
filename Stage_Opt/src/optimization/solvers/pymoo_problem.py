"""PyMOO problem definition for rocket stage optimization."""
import numpy as np
from pymoo.core.problem import Problem
from ...utils.config import logger
from ..objective import objective_with_penalty
from ..cache import OptimizationCache

def tournament_comp(pop, P, **kwargs):
    """Tournament selection comparator."""
    S = np.full(P.shape[0], np.nan)

    for i in range(P.shape[0]):
        a, b = P[i, 0], P[i, 1]

        # Get fitness values using new pymoo API
        a_fitness = pop[a].get("F")[0]
        b_fitness = pop[b].get("F")[0]
        
        # Get constraint violations
        a_cv = pop[a].get("CV")[0] if pop[a].get("CV") is not None else 0.0
        b_cv = pop[b].get("CV")[0] if pop[b].get("CV") is not None else 0.0

        # If invalid solutions exist, always prefer valid solution
        if a_cv > 0.0 or b_cv > 0.0:
            if a_cv < b_cv:
                S[i] = a
            else:
                S[i] = b
        # Otherwise use objective values
        else:
            if a_fitness < b_fitness:
                S[i] = a
            else:
                S[i] = b

    return S

class RocketStageProblem(Problem):
    """Problem definition for rocket stage optimization."""
    
    def __init__(self, solver, n_var, bounds):
        """Initialize problem."""
        bounds = np.array(bounds)  # Convert to numpy array if not already
        super().__init__(
            n_var=n_var,
            n_obj=1,
            n_constr=2,  # Adding explicit constraints instead of penalties
            xl=bounds[:, 0],  # Lower bounds
            xu=bounds[:, 1]   # Upper bounds
        )
        self.solver = solver
        
        # Create a simplified cache filename
        solver_type = "adaptive_ga" if "adaptive" in solver.name.lower() else "ga"
        self.cache = OptimizationCache(
            cache_file=f"{solver_type}_cache.pkl",
            max_size=10000
        )
        
        # Initialize evaluation counters
        self.total_evals = 0
        self.cache_hits = 0
        self.best_fitness = float('inf')
        self.worst_fitness = float('-inf')
        
    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate objective function with caching."""
        try:
            population_size = len(x)
            f = np.zeros(population_size)
            g = np.zeros((population_size, 2))  # Constraint violations
            
            for i, x_i in enumerate(x):
                try:
                    # Detailed logging for debugging
                    logger.debug(f"Evaluating individual {i} with parameters: {x_i}")
                    
                    # Calculate objective and constraints
                    f[i], g1, g2 = objective_with_penalty(
                        dv=x_i,
                        G0=self.solver.G0,
                        ISP=self.solver.ISP,
                        EPSILON=self.solver.EPSILON,
                        TOTAL_DELTA_V=self.solver.TOTAL_DELTA_V,
                        return_constraints=True  # Modified objective function to return constraints
                    )
                    
                    g[i] = [g1, g2]
                    
                    # Cache only valid solutions
                    if g1 <= 0 and g2 <= 0:
                        x_tuple = tuple(np.round(x_i, decimals=6))  # Round to reduce floating point issues
                        self.cache.set(x_tuple, f[i])
                    
                    # Update statistics
                    if g1 <= 0 and g2 <= 0:  # Only update stats for valid solutions
                        self.best_fitness = min(self.best_fitness, f[i])
                        self.worst_fitness = max(self.worst_fitness, f[i])
                    
                except Exception as ind_error:
                    logger.error(f"Error evaluating individual {i}: {str(ind_error)}")
                    f[i] = float('inf')
                    g[i] = [float('inf'), float('inf')]
            
            self.total_evals += population_size
            
            # Log evaluation statistics periodically
            if self.total_evals % 100 == 0:
                valid_solutions = np.sum(np.all(g <= 0, axis=1))
                logger.info(f"Problem Statistics:")
                logger.info(f"  Total Evaluations: {self.total_evals}")
                logger.info(f"  Valid Solutions: {valid_solutions}/{population_size}")
                if valid_solutions > 0:
                    logger.info(f"  Best Fitness (valid): {self.best_fitness:.6f}")
                
            out["F"] = f
            out["G"] = g  # Explicit constraint violations
            
        except Exception as e:
            logger.error(f"Critical error in problem evaluation: {str(e)}", exc_info=True)
            out["F"] = np.array([float('inf')] * len(x))
            out["G"] = np.array([[float('inf'), float('inf')]] * len(x))
