"""Genetic Algorithm solver implementation."""
import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.mutation.pm import PM
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize
from ...utils.config import logger
from .base_solver import BaseSolver
from .pymoo_problem import RocketStageProblem, objective_with_penalty

class GeneticAlgorithmSolver(BaseSolver):
    """Genetic Algorithm solver implementation."""
    
    def __init__(self, config, problem_params):
        """Initialize GA solver."""
        super().__init__(config, problem_params)
        self.solver_specific = self.solver_config.get('solver_specific', {})
        
    def solve(self, initial_guess, bounds):
        """Solve using genetic algorithm."""
        try:
            logger.info("Starting GA optimization...")
            
            # Setup problem
            n_var = len(initial_guess)
            bounds = np.array(bounds)  # Convert bounds to numpy array
            problem = RocketStageProblem(
                solver=self,
                n_var=n_var,
                bounds=bounds
            )
            
            # Get solver parameters from config
            pop_size = int(self.solver_specific.get('pop_size', 100))
            n_gen = int(self.solver_specific.get('n_generations', 100))
            crossover_prob = float(self.solver_specific.get('crossover_prob', 0.9))
            mutation_prob = float(self.solver_specific.get('mutation_prob', 0.1))
            tournament_size = int(self.solver_specific.get('tournament_size', 3))
            
            # Setup algorithm
            algorithm = GA(
                pop_size=pop_size,
                sampling=FloatRandomSampling(),
                crossover=SBX(prob=crossover_prob, eta=30),
                mutation=PM(prob=mutation_prob, eta=30),
                selection=TournamentSelection(pressure=tournament_size),
                eliminate_duplicates=True
            )
            
            # Run optimization
            res = minimize(
                problem,
                algorithm,
                ('n_gen', n_gen),
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
