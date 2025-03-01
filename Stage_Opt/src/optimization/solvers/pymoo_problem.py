"""Common PyMOO problem definitions and utilities."""
import numpy as np
from pymoo.core.problem import Problem
from pymoo.operators.selection.tournament import TournamentSelection

def tournament_comp(pop, P, **kwargs):
    """Tournament selection comparison function.
    
    Args:
        pop: Population to select from
        P: Tournament pairs matrix
        **kwargs: Additional arguments
        
    Returns:
        np.ndarray: Selected individual indices
    """
    S = np.full(P.shape[0], np.nan)
    
    for i in range(P.shape[0]):
        a, b = P[i, 0], P[i, 1]
        
        # Get individuals from population
        ind_a, ind_b = pop[a], pop[b]
        
        # Get constraint violations (if any)
        a_cv = 0 if ind_a.CV is None else float(ind_a.CV[0])
        b_cv = 0 if ind_b.CV is None else float(ind_b.CV[0])
        
        # Get objective values
        a_f = float(ind_a.F[0])
        b_f = float(ind_b.F[0])
        
        # If both feasible or both infeasible
        if (a_cv <= 0 and b_cv <= 0) or (a_cv > 0 and b_cv > 0):
            S[i] = a if a_f < b_f else b
        # If one is feasible and other is not
        else:
            S[i] = a if a_cv <= 0 else b
            
    return S

class RocketStageProblem(Problem):
    """Problem definition for pymoo-based solvers."""
    
    def __init__(self, solver, n_var, bounds):
        """Initialize problem.
        
        Args:
            solver: Solver instance
            n_var: Number of variables
            bounds: List of (min, max) bounds for each variable
        """
        super().__init__(
            n_var=n_var,
            n_obj=1,
            n_constr=1,
            xl=np.array([b[0] for b in bounds]),
            xu=np.array([b[1] for b in bounds])
        )
        self.solver = solver
        
    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate solutions.
        
        Args:
            x: Solutions to evaluate
            out: Output dictionary
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        # Evaluate each solution
        f = np.zeros((x.shape[0], 1))
        g = np.zeros((x.shape[0], 1))  # Constraints
        
        for i in range(x.shape[0]):
            # Calculate objective (negative since pymoo minimizes)
            f[i, 0] = -self.solver.objective_with_penalty(x[i])
            
            # Calculate constraint violation
            stage_ratios, _ = self.solver.calculate_stage_ratios(x[i])
            delta_v = self.solver.calculate_delta_v(stage_ratios)
            total_dv = np.sum(delta_v)
            g[i, 0] = total_dv - self.solver.TOTAL_DELTA_V
        
        out["F"] = f
        out["G"] = g
