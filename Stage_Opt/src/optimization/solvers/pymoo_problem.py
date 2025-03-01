"""PyMOO problem definition for rocket stage optimization."""
import numpy as np
from pymoo.core.problem import Problem
from ...utils.config import logger
from ..objective import objective_with_penalty

def tournament_comp(pop, P, **kwargs):
    """Tournament selection comparator.
    
    Args:
        pop: Population object containing individuals
        P: Matrix of shape (n_tournaments, 2) containing indices of individuals to compare
        kwargs: Additional keyword arguments passed by pymoo
        
    Returns:
        S: Array containing selected individual indices
    """
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
        """Initialize problem.
        
        Args:
            solver: Solver instance containing problem parameters
            n_var: Number of variables (stages)
            bounds: Variable bounds as numpy array with shape (n_var, 2)
        """
        bounds = np.array(bounds)  # Convert to numpy array if not already
        super().__init__(
            n_var=n_var,
            n_obj=1,
            n_constr=0,
            xl=bounds[:, 0],  # Lower bounds
            xu=bounds[:, 1]   # Upper bounds
        )
        self.solver = solver
        
    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate objective function."""
        try:
            f = np.array([
                objective_with_penalty(
                    dv=x_i,
                    G0=self.solver.G0,
                    ISP=self.solver.ISP,
                    EPSILON=self.solver.EPSILON,
                    TOTAL_DELTA_V=self.solver.TOTAL_DELTA_V
                )
                for x_i in x
            ])
            out["F"] = f
            
        except Exception as e:
            logger.error(f"Error in problem evaluation: {str(e)}")
            out["F"] = np.array([float('inf')] * len(x))
