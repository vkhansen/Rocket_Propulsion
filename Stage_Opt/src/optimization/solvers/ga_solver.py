"""Genetic Algorithm solver implementation."""
import numpy as np
import time
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize
from .base_solver import BaseSolver
from .pymoo_problem import RocketStageProblem, tournament_comp
from ...utils.config import logger

class GeneticAlgorithmSolver(BaseSolver):
    """Genetic Algorithm solver implementation."""
    
    def __init__(self, config, problem_params):
        """Initialize GA solver."""
        super().__init__(config, problem_params)  # Initialize base solver first
        self.solver_specific = self.solver_config.get('solver_specific', {})
        
    def solve(self, initial_guess, bounds):
        """Solve using Genetic Algorithm.
        
        Args:
            initial_guess: Initial solution guess
            bounds: List of (min, max) bounds for each variable
            
        Returns:
            dict: Optimization results
        """
        try:
            logger.info("Starting GA optimization...")
            start_time = time.time()
            
            # Get solver parameters from config
            ga_config = self.solver_config.get('solver_specific', {})
            pop_size = ga_config.get('pop_size', 100)
            n_gen = ga_config.get('n_generations', 100)
            tournament_size = ga_config.get('tournament_size', 3)
            crossover_prob = ga_config.get('crossover_prob', 0.9)
            mutation_prob = ga_config.get('mutation_prob', 0.1)
            
            # Initialize problem
            problem = RocketStageProblem(self, len(initial_guess), bounds)
            
            # Initialize algorithm with specific operators
            algorithm = GA(
                pop_size=pop_size,
                selection=TournamentSelection(
                    pressure=tournament_size,
                    func_comp=tournament_comp
                ),
                crossover=SBX(
                    prob=crossover_prob,
                    eta=15
                ),
                mutation=PM(
                    prob=mutation_prob,
                    eta=20
                ),
                eliminate_duplicates=True
            )
            
            # Run optimization
            result = minimize(
                problem,
                algorithm,
                ('n_gen', n_gen),
                seed=42,
                verbose=False
            )
            
            execution_time = time.time() - start_time
            
            if result.success:
                return self.process_results(
                    result.X,
                    success=True,
                    n_iter=result.algorithm.n_iter,
                    n_evals=result.algorithm.evaluator.n_eval,
                    time=execution_time
                )
            else:
                return self.process_results(
                    np.zeros_like(initial_guess),
                    success=False,
                    message="GA optimization failed to converge",
                    time=execution_time
                )
            
        except Exception as e:
            logger.error(f"Error in GA solver: {str(e)}")
            return self.process_results(
                np.zeros_like(initial_guess),
                success=False,
                message=f"Error in GA solver: {str(e)}"
            )
