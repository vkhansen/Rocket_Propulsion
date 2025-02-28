"""Genetic Algorithm solver implementation."""
import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from .base_solver import BaseSolver
from utils.config import logger

class RocketStageProblem(Problem):
    """Problem definition for pymoo GA."""
    
    def __init__(self, solver, n_var, bounds):
        super().__init__(
            n_var=n_var,
            n_obj=1,
            n_constr=0,
            xl=np.array([b[0] for b in bounds]),
            xu=np.array([b[1] for b in bounds])
        )
        self.solver = solver
        
    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate solutions."""
        f = np.zeros(len(x))
        for i, xi in enumerate(x):
            penalty = self.solver.enforce_constraints(xi)
            payload_fraction = -self.solver.calculate_fitness(xi)
            penalty_coeff = self.solver.solver_config.get('penalty_coefficient', 1e3)
            f[i] = payload_fraction + penalty_coeff * penalty
        out["F"] = f

class GASolver(BaseSolver):
    """Genetic Algorithm solver implementation."""
    
    def __init__(self, config, problem_params):
        """Initialize GA solver."""
        super().__init__(config, problem_params)
        self.solver_specific = self.solver_config.get('solver_specific', {})
        
    def solve(self, initial_guess, bounds):
        """Solve using Genetic Algorithm."""
        try:
            # Get solver parameters
            pop_size = self.solver_specific.get('population_size', 100)
            n_gen = self.solver_specific.get('n_generations', 200)
            mutation = self.solver_specific.get('mutation', {
                'eta': 20,
                'prob': 0.1
            })
            crossover = self.solver_specific.get('crossover', {
                'eta': 15,
                'prob': 0.8
            })
            
            # Create problem
            problem = RocketStageProblem(self, len(initial_guess), bounds)
            
            # Initialize algorithm
            algorithm = GA(
                pop_size=pop_size,
                mutation=mutation,
                crossover=crossover
            )
            
            # Run optimization
            result = minimize(
                problem,
                algorithm,
                ('n_gen', n_gen),
                seed=1,
                verbose=False
            )
            
            # Get best solution
            x = result.X
            payload_fraction = self.calculate_fitness(x)
            stage_ratios, stage_info = calculate_stage_ratios(
                x, self.G0, self.ISP, self.EPSILON
            )
            
            return {
                'success': True,
                'message': "Optimization completed",
                'payload_fraction': payload_fraction,
                'stages': stage_info,
                'n_iterations': n_gen,
                'n_function_evals': result.algorithm.evaluator.n_eval
            }
            
        except Exception as e:
            logger.error(f"Error in GA solver: {str(e)}")
            return {
                'success': False,
                'message': f"Error: {str(e)}",
                'payload_fraction': 0.0,
                'stages': [],
                'n_iterations': 0,
                'n_function_evals': 0
            }
