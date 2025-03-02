"""PyMOO problem definition for rocket stage optimization."""
import numpy as np
from pymoo.core.problem import Problem
from src.utils.config import logger
from src.optimization.objective import objective_with_penalty, get_constraint_violations
from src.optimization.cache import OptimizationCache
from datetime import datetime

def tournament_comp(pop, P, **kwargs):
    """Tournament selection comparator."""
    try:
        # Initialize selection array with first individual from each tournament
        S = P[:, 0].copy()  # Default to first individual
        
        for i in range(P.shape[0]):
            try:
                # Get indices for this tournament
                a, b = P[i, 0], P[i, 1]
                
                # Validate population members
                if not all(isinstance(ind, type(pop[0])) for ind in [pop[a], pop[b]]):
                    continue
                    
                # Get fitness values safely
                a_F = getattr(pop[a], "F", None)
                b_F = getattr(pop[b], "F", None)
                
                if a_F is None or b_F is None:
                    continue
                    
                # Get constraint violations safely
                a_G = getattr(pop[a], "G", None)
                b_G = getattr(pop[b], "G", None)
                
                # Calculate constraint violations
                a_cv = float(np.sum(np.maximum(a_G, 0))) if a_G is not None else float('inf')
                b_cv = float(np.sum(np.maximum(b_G, 0))) if b_G is not None else float('inf')
                
                # Compare solutions
                if a_cv == 0 and b_cv == 0:  # Both feasible
                    if a_F[0] < b_F[0]:  # Minimize
                        S[i] = a
                    else:
                        S[i] = b
                else:  # At least one infeasible
                    if a_cv < b_cv:
                        S[i] = a
                    else:
                        S[i] = b
                        
            except Exception as e:
                logger.error(f"Error in tournament {i}: {str(e)}")
                # Keep default selection (first individual)
                continue
                
        return S
        
    except Exception as e:
        logger.error(f"Error in tournament selection: {str(e)}")
        # Return first individuals as fallback
        return P[:, 0].copy()

class RocketStageProblem(Problem):
    """Problem definition for rocket stage optimization."""
    
    def __init__(self, solver, n_var, bounds):
        """Initialize problem."""
        bounds = np.array(bounds)  # Convert to numpy array if not already
        super().__init__(
            n_var=n_var,
            n_obj=1,
            n_constr=2,  # Two constraints: delta-v and physical constraints
            lower=bounds[:, 0],  # Lower bounds
            upper=bounds[:, 1],   # Upper bounds
            vtype=float
        )
        self.solver = solver
        self.cache = OptimizationCache()
        
        # Initialize evaluation counters and statistics
        self.total_evals = 0
        self.cache_hits = 0
        self.best_objective = float('inf')
        self.best_feasible = None
        self.constraint_violations = []
        
        # Initialize constraint thresholds
        self.feasibility_threshold = 1e-4  # Threshold for considering a solution feasible
        self.constraint_history = []  # Track constraint violations over time
        
    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate objective function with caching.
        
        Args:
            x: Input matrix of shape (n_individuals, n_variables)
            out: Output dictionary containing objective and constraint values
        """
        try:
            if x is None or not isinstance(x, np.ndarray):
                raise ValueError("Invalid input array")
                
            n_individuals = x.shape[0]
            F = np.zeros((n_individuals, 1))  # Objective values
            G = np.zeros((n_individuals, 2))  # Constraint violations
            
            for i in range(n_individuals):
                try:
                    x_i = x[i]
                    if not np.all(np.isfinite(x_i)):
                        raise ValueError(f"Non-finite values in individual {i}")
                        
                    x_tuple = tuple(x_i)
                    
                    # Try to get from cache first
                    cached_result = self.cache.get(x_tuple)
                    if cached_result is not None:
                        self.cache_hits += 1
                        objective, dv_constraint, physical_constraint = cached_result
                    else:
                        # Calculate objective and constraints
                        objective, dv_constraint, physical_constraint = objective_with_penalty(
                            dv=x_i,
                            G0=self.solver.G0,
                            ISP=self.solver.ISP,
                            EPSILON=self.solver.EPSILON,
                            TOTAL_DELTA_V=self.solver.TOTAL_DELTA_V,
                            return_tuple=True  # Always get tuple for PyMOO
                        )
                        self.cache.add(x_tuple, (objective, dv_constraint, physical_constraint))
                    
                    # Store results for this individual
                    F[i, 0] = objective
                    G[i, 0] = dv_constraint
                    G[i, 1] = physical_constraint
                    
                    # Update statistics
                    self.total_evals += 1
                    
                    # Track constraint violations with relative scaling
                    self.constraint_violations.append((dv_constraint, physical_constraint))
                    if len(self.constraint_violations) > 1000:  # Keep last 1000 evaluations
                        self.constraint_violations.pop(0)
                    
                    # Update best feasible solution
                    if max(dv_constraint, physical_constraint) <= self.feasibility_threshold:
                        if -objective < self.best_objective:  # Note: objective is negative for maximization
                            self.best_objective = -objective
                            self.best_feasible = x_i.copy()
                            
                except Exception as e:
                    logger.error(f"Error evaluating individual {i}: {str(e)}")
                    # Penalize this individual
                    F[i, 0] = 1e6
                    G[i, 0] = 1e6
                    G[i, 1] = 1e6
            
            # Store results
            out["F"] = F  # Objective values (to be minimized)
            out["G"] = G  # Constraints g(x) <= 0
            
            # Update constraint history for adaptive handling
            mean_violations = np.mean(G, axis=0)
            self.constraint_history.append(mean_violations)
            if len(self.constraint_history) > 100:  # Keep last 100 generations
                self.constraint_history.pop(0)
            
        except Exception as e:
            logger.error(f"Error in evaluation: {str(e)}")
            # Return highly penalized solutions for all individuals
            n_individuals = x.shape[0] if isinstance(x, np.ndarray) else 1
            out["F"] = np.full((n_individuals, 1), 1e6)
            out["G"] = np.full((n_individuals, 2), 1e6)

    def get_constraint_statistics(self):
        """Get statistics about constraint violations."""
        if not self.constraint_violations:
            return {
                "dv_mean": 0.0, 
                "dv_max": 0.0, 
                "physical_mean": 0.0, 
                "physical_max": 0.0,
                "feasible_solutions": 0,
                "total_evaluations": self.total_evals
            }
            
        dv_violations, physical_violations = zip(*self.constraint_violations)
        feasible_count = sum(1 for dv, phys in self.constraint_violations 
                           if max(dv, phys) <= self.feasibility_threshold)
        
        return {
            "dv_mean": float(np.mean(dv_violations)),
            "dv_max": float(np.max(dv_violations)),
            "physical_mean": float(np.mean(physical_violations)),
            "physical_max": float(np.max(physical_violations)),
            "feasible_solutions": feasible_count,
            "total_evaluations": self.total_evals
        }
