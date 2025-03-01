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
        super().__init__(config, problem_params)
        self.solver_specific = self.solver_config.get('solver_specific', {})
        
        # Initialize solver parameters with defaults
        self.pop_size = int(self.solver_specific.get('population_size', 100))
        self.n_gen = int(self.solver_specific.get('n_generations', 200))
        self.tournament_size = int(self.solver_specific.get('tournament_size', 3))
        self.crossover_rate = float(self.solver_specific.get('crossover_rate', 0.9))
        self.mutation_rate = float(self.solver_specific.get('mutation_rate', 0.1))
        
        logger.debug(f"Initialized {self.name} with parameters: "
                    f"pop_size={self.pop_size}, n_gen={self.n_gen}, "
                    f"tournament_size={self.tournament_size}, "
                    f"crossover_rate={self.crossover_rate}, "
                    f"mutation_rate={self.mutation_rate}")
    
    def solve(self, initial_guess, bounds):
        """Solve using Genetic Algorithm."""
        try:
            logger.info(f"Starting {self.name} optimization")
            logger.debug(f"Initial guess: {initial_guess}")
            logger.debug(f"Bounds: {bounds}")
            start_time = time.time()
            
            # Initialize problem
            problem = RocketStageProblem(
                solver=self,
                n_var=len(initial_guess),
                bounds=bounds
            )
            logger.debug("Problem initialized")
            
            # Initialize algorithm with specific operators
            algorithm = GA(
                pop_size=self.pop_size,
                sampling=np.array(initial_guess).reshape(1, -1),
                selection=TournamentSelection(
                    pressure=self.tournament_size,
                    func_comp=tournament_comp
                ),
                crossover=SBX(
                    prob=self.crossover_rate,
                    eta=15
                ),
                mutation=PM(
                    prob=self.mutation_rate,
                    eta=20
                ),
                eliminate_duplicates=True
            )
            logger.debug(f"Algorithm initialized with parameters: "
                        f"pop_size={self.pop_size}, tournament_size={self.tournament_size}, "
                        f"crossover_rate={self.crossover_rate}, mutation_rate={self.mutation_rate}")
            
            # Run optimization
            logger.info("Starting optimization process...")
            result = minimize(
                problem,
                algorithm,
                ('n_gen', self.n_gen),
                seed=42,
                verbose=False
            )
            
            # Process results
            best_x = result.X.astype(float)  # Convert to native float
            best_f = float(result.F[0])  # Convert to native float
            
            # Log optimization results
            logger.info(f"Optimization completed after {result.algorithm.n_gen} generations")
            logger.info(f"Number of function evaluations: {result.algorithm.evaluator.n_eval}")
            logger.info(f"Best fitness achieved: {best_f:.6f}")
            
            # Calculate final stage ratios and create results
            stage_ratios, mass_ratios = self.calculate_stage_ratios(best_x)
            stages = self.create_stage_results(best_x, stage_ratios)
            logger.debug(f"Stage ratios: {stage_ratios}")
            logger.debug(f"Mass ratios: {mass_ratios}")
            
            execution_time = time.time() - start_time
            logger.info(f"Optimization completed in {execution_time:.2f} seconds")
            
            return {
                'x': best_x.tolist(),
                'f': best_f,
                'success': True,
                'stages': stages,
                'execution_time': execution_time,
                'n_generations': result.algorithm.n_gen,
                'n_evaluations': result.algorithm.evaluator.n_eval
            }
            
        except Exception as e:
            logger.error(f"Error in {self.name}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
