"""Genetic Algorithm solver implementation."""
import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from .base_solver import BaseSolver
from ...utils.config import logger

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
        """Evaluate the objective function."""
        # Convert to list of solutions if only one solution
        if len(x.shape) == 1:
            x = [x]
            
        # Evaluate each solution
        f = []
        for solution in x:
            fitness = self.solver.objective_with_penalty(solution)
            f.append(fitness)
            
        out["F"] = np.array(f)

class GeneticAlgorithmSolver(BaseSolver):
    """Genetic Algorithm solver implementation."""
    
    def __init__(self, config, problem_params):
        """Initialize GA solver."""
        super().__init__(config, problem_params)
        self.solver_specific = self.solver_config.get('solver_specific', {})
        
    def solve(self):
        """Solve using Genetic Algorithm."""
        try:
            logger.info("Starting Genetic Algorithm optimization...")
            
            # Create problem instance
            problem = RocketStageProblem(
                solver=self,
                n_var=len(self.initial_guess),
                bounds=self.bounds
            )
            
            # Configure GA parameters
            algorithm = GA(
                pop_size=self.solver_specific.get('population_size', 50),
                eliminate_duplicates=True
            )
            
            # Run optimization
            res = minimize(
                problem,
                algorithm,
                ('n_gen', self.solver_specific.get('n_generations', 100)),
                seed=42,
                verbose=False
            )
            
            # Process results
            if res.X is not None:
                stage_ratios, mass_ratios = self.calculate_stage_ratios(res.X)
                payload_fraction = self.calculate_payload_fraction(stage_ratios)
                
                result = {
                    'success': True,
                    'x': res.X.tolist(),
                    'fun': float(res.F[0]),
                    'payload_fraction': payload_fraction,
                    'stage_ratios': stage_ratios.tolist(),
                    'mass_ratios': mass_ratios.tolist(),
                    'stages': self.create_stage_results(res.X, stage_ratios),
                    'execution_time': res.exec_time
                }
            else:
                result = {
                    'success': False,
                    'message': "GA optimization failed to find a solution"
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in GA solver: {str(e)}")
            return {
                'success': False,
                'message': f"Error in GA solver: {str(e)}"
            }
