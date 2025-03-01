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
            n_constr=0,
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
            f = np.zeros(len(x))
            for i, x_i in enumerate(x):
                # Convert numpy array to tuple for caching
                x_tuple = tuple(x_i)
                
                # Try to get from cache first
                cached_value = self.cache.get(x_tuple)
                if cached_value is not None:
                    f[i] = cached_value
                    self.cache_hits += 1
                else:
                    # Calculate and cache if not found
                    f[i] = objective_with_penalty(
                        dv=x_i,
                        G0=self.solver.G0,
                        ISP=self.solver.ISP,
                        EPSILON=self.solver.EPSILON,
                        TOTAL_DELTA_V=self.solver.TOTAL_DELTA_V
                    )
                    self.cache.set(x_tuple, f[i])
                
                # Update statistics
                self.best_fitness = min(self.best_fitness, f[i])
                self.worst_fitness = max(self.worst_fitness, f[i])
            
            self.total_evals += len(x)
            
            # Log evaluation statistics periodically
            if self.total_evals % 100 == 0:
                cache_rate = (self.cache_hits / self.total_evals) * 100
                logger.debug(f"Problem Statistics:")
                logger.debug(f"  Total Evaluations: {self.total_evals}")
                logger.debug(f"  Cache Hit Rate: {cache_rate:.1f}%")
                logger.debug(f"  Best Fitness: {self.best_fitness:.6f}")
                logger.debug(f"  Worst Fitness: {self.worst_fitness:.6f}")
                logger.debug(f"  Fitness Range: {self.worst_fitness - self.best_fitness:.6f}")
            
            out["F"] = f
            
        except Exception as e:
            logger.error(f"Error in problem evaluation: {str(e)}")
            out["F"] = np.array([float('inf')] * len(x))
