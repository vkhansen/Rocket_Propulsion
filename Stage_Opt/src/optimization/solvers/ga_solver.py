"""Genetic Algorithm solver implementation."""
import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.mutation.pm import PM
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from ...utils.config import logger
from .base_solver import BaseSolver
from .pymoo_problem import RocketStageProblem, tournament_comp
from ..objective import objective_with_penalty

class GeneticAlgorithmSolver(BaseSolver):
    """Genetic Algorithm solver implementation."""
    
    def solve(self, initial_guess, bounds):
        """Solve using genetic algorithm."""
        try:
            logger.info("Starting GA optimization...")
            
            # Setup problem
            n_var = len(initial_guess)
            problem = RocketStageProblem(
                solver=self,
                n_var=n_var,
                bounds=bounds
            )
            
            # Setup algorithm
            algorithm = GA(
                pop_size=50,
                sampling=FloatRandomSampling(),
                crossover=SBX(prob=0.9, eta=30),
                mutation=PM(prob=0.1, eta=30),
                eliminate_duplicates=True,
                selection=tournament_comp
            )
            
            # Run optimization
            res = minimize(
                problem,
                algorithm,
                seed=1,
                verbose=False
            )
            
            # Process results
            success = res.success if hasattr(res, 'success') else True
            message = res.message if hasattr(res, 'message') else ""
            x = res.X if hasattr(res, 'X') else initial_guess
            n_gen = res.algorithm.n_gen if hasattr(res.algorithm, 'n_gen') else 0
            n_eval = res.algorithm.evaluator.n_eval if hasattr(res.algorithm, 'evaluator') else 0
            
            return self.process_results(
                x=x,
                success=success,
                message=message,
                n_iterations=n_gen,
                n_function_evals=n_eval,
                time=0.0  # Time not tracked by pymoo
            )
            
        except Exception as e:
            logger.error(f"Error in GA solver: {str(e)}")
            return self.process_results(
                x=initial_guess,
                success=False,
                message=str(e)
            )
