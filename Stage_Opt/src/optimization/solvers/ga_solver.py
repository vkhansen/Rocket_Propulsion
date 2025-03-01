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
            logger.debug(f"Initial guess: {initial_guess}")
            logger.debug(f"Bounds: {bounds}")
            start_time = time.time()
            
            # Get solver parameters from config
            ga_config = self.solver_config.get('solver_specific', {})
            pop_size = ga_config.get('pop_size', 100)
            n_gen = ga_config.get('n_generations', 100)
            tournament_size = ga_config.get('tournament_size', 3)
            crossover_prob = ga_config.get('crossover_prob', 0.9)
            mutation_prob = ga_config.get('mutation_prob', 0.1)
            
            logger.debug(f"GA parameters: pop_size={pop_size}, n_gen={n_gen}, "
                      f"tournament_size={tournament_size}, crossover_prob={crossover_prob}, "
                      f"mutation_prob={mutation_prob}")
            
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
            logger.debug(f"GA optimization completed in {execution_time:.2f} seconds")
            
            # Check solution validity
            if result.X is None:
                logger.warning("GA optimization failed to find any solution")
                return self.process_results(
                    np.zeros_like(initial_guess),
                    success=False,
                    message="GA optimization failed to find any solution",
                    time=execution_time
                )
                
            # Check constraint violation
            if result.CV is not None and float(result.CV.min()) > 1e-6:
                logger.warning("GA optimization failed to find a feasible solution")
                return self.process_results(
                    np.zeros_like(initial_guess),
                    success=False,
                    message="GA optimization failed to find a feasible solution",
                    time=execution_time
                )
            
            # Get best solution
            best_x = result.X
            if isinstance(best_x, np.ndarray):
                best_x = best_x.copy()  # Make a copy to avoid reference issues
            else:
                best_x = np.array(best_x)
                
            logger.info("Successfully completed GA optimization")
            return self.process_results(
                best_x,
                success=True,
                n_iter=result.algorithm.n_iter,
                n_evals=result.algorithm.evaluator.n_eval,
                time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Error in GA solver: {str(e)}")
            return self.process_results(
                np.zeros_like(initial_guess),
                success=False,
                message=f"Error in GA solver: {str(e)}"
            )
