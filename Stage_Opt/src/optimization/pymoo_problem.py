"""PyMOO problem definition for rocket stage optimization."""
import numpy as np
from pymoo.core.problem import Problem
from src.utils.config import logger
from src.optimization.objective import objective_with_penalty
from src.optimization.cache import OptimizationCache
from datetime import datetime

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
            n_constr=2,  # Two constraints: delta-v and physical constraints
            xl=bounds[:, 0],  # Lower bounds
            xu=bounds[:, 1]   # Upper bounds
        )
        self.solver = solver
        
        # Create a simplified cache filename with timestamp to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        solver_type = "adaptive_ga" if "adaptive" in solver.name.lower() else "ga"
        self.cache = OptimizationCache(
            cache_file=f"{solver_type}_cache_{timestamp}.pkl",
            max_size=10000
        )
        
        # Initialize evaluation counters and statistics
        self.total_evals = 0
        self.cache_hits = 0
        self.best_objective = float('inf')
        self.best_feasible = None
        self.constraint_violations = []
        
    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate objective function with caching.
        
        Args:
            x: Input vector (delta-v values)
            out: Output dictionary containing objective and constraint values
        """
        try:
            # Convert to tuple for hashing in cache
            x_tuple = tuple(x.flatten())
            
            # Try to get from cache first
            cached_result = self.cache.get(x_tuple)
            if cached_result is not None:
                self.cache_hits += 1
                objective, dv_constraint, physical_constraint = cached_result
            else:
                # Calculate objective and constraints
                objective, dv_constraint, physical_constraint = self.solver.objective_with_penalty(x)
                self.cache.add(x_tuple, (objective, dv_constraint, physical_constraint))
            
            # Update statistics
            self.total_evals += 1
            if dv_constraint <= 0 and physical_constraint <= 0:  # Feasible solution
                if -objective < self.best_objective:  # Note: objective is negative for maximization
                    self.best_objective = -objective
                    self.best_feasible = x.copy()
            
            # Track constraint violations for adaptive penalty adjustment
            self.constraint_violations.append((dv_constraint, physical_constraint))
            if len(self.constraint_violations) > 1000:  # Keep last 1000 evaluations
                self.constraint_violations.pop(0)
            
            # Store results
            out["F"] = np.array([objective])  # Objective value (to be minimized)
            out["G"] = np.array([dv_constraint, physical_constraint])  # Constraints g(x) <= 0
            
        except Exception as e:
            logger.error(f"Error in evaluation: {str(e)}")
            # Return a highly penalized solution
            out["F"] = np.array([1e6])
            out["G"] = np.array([1e6, 1e6])
            
    def get_constraint_statistics(self):
        """Get statistics about constraint violations."""
        if not self.constraint_violations:
            return {"dv_mean": 0.0, "dv_max": 0.0, "physical_mean": 0.0, "physical_max": 0.0}
            
        dv_violations, physical_violations = zip(*self.constraint_violations)
        return {
            "dv_mean": np.mean(dv_violations),
            "dv_max": np.max(dv_violations),
            "physical_mean": np.mean(physical_violations),
            "physical_max": np.max(physical_violations)
        }
