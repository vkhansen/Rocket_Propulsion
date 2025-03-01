"""Adaptive Genetic Algorithm solver implementation."""
import numpy as np
import time
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.callback import Callback
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize
from .base_solver import BaseSolver
from .pymoo_problem import RocketStageProblem, tournament_comp
from ...utils.config import logger

class AdaptiveCallback(Callback):
    """Callback for adaptive population size and operator parameters."""
    
    def __init__(self, solver, stagnation_limit=10, min_mutation_rate=0.01, max_mutation_rate=0.5, 
                 min_crossover_rate=0.5, max_crossover_rate=1.0, elite_size=2, convergence_threshold=1e-6):
        """Initialize adaptive callback.
        
        Args:
            solver: Solver instance
            stagnation_limit: Number of generations without improvement before adaptation
            min_mutation_rate: Minimum mutation rate
            max_mutation_rate: Maximum mutation rate
            min_crossover_rate: Minimum crossover rate
            max_crossover_rate: Maximum crossover rate
            elite_size: Number of elite solutions to preserve
            convergence_threshold: Threshold for convergence detection
        """
        super().__init__()
        self.solver = solver
        self.stagnation_limit = stagnation_limit
        self.min_mutation_rate = min_mutation_rate
        self.max_mutation_rate = max_mutation_rate
        self.min_crossover_rate = min_crossover_rate
        self.max_crossover_rate = max_crossover_rate
        self.elite_size = elite_size
        self.convergence_threshold = convergence_threshold
        self.stagnation_counter = 0
        self.best_fitness = -np.inf
        self.diversity_history = []
        
    def calculate_diversity(self, pop):
        """Calculate population diversity."""
        X = np.array([ind.X for ind in pop])  # Get decision variables as numpy array
        return np.mean(np.std(X, axis=0))
        
    def notify(self, algorithm):
        """Adapt parameters based on optimization progress."""
        # Get current best fitness (negative since we're minimizing)
        current_best = -float(algorithm.opt.F[0])
        
        # Calculate diversity
        diversity = self.calculate_diversity(algorithm.pop)
        self.diversity_history.append(diversity)
        
        # Check for improvement
        if current_best > self.best_fitness:
            self.best_fitness = current_best
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
        
        # Adapt population size
        if self.stagnation_counter > self.stagnation_limit:
            algorithm.pop_size = min(int(algorithm.pop_size * 1.5), 500)
            self.stagnation_counter = 0
        
        # Adapt operator parameters based on diversity
        if len(self.diversity_history) > 2:
            if diversity < np.mean(self.diversity_history[-3:]):
                # Increase mutation probability to promote exploration
                algorithm.mating.mutation.prob = min(
                    algorithm.mating.mutation.prob * 1.2, self.max_mutation_rate
                )
            else:
                # Decrease mutation probability to promote exploitation
                algorithm.mating.mutation.prob = max(
                    algorithm.mating.mutation.prob * 0.8, self.min_mutation_rate
                )
            
            # Adapt crossover probability based on convergence
            if abs(current_best - self.best_fitness) < self.convergence_threshold:
                # Increase crossover probability to promote convergence
                algorithm.mating.crossover.prob = min(
                    algorithm.mating.crossover.prob * 1.1, self.max_crossover_rate
                )
            else:
                # Decrease crossover probability to promote exploration
                algorithm.mating.crossover.prob = max(
                    algorithm.mating.crossover.prob * 0.9, self.min_crossover_rate
                )

class AdaptiveGeneticAlgorithmSolver(BaseSolver):
    """Adaptive Genetic Algorithm solver implementation."""
    
    def __init__(self, config, problem_params):
        """Initialize Adaptive GA solver."""
        super().__init__(config, problem_params)  # Initialize base solver first
        self.solver_specific = self.solver_config.get('solver_specific', {})
        
    def solve(self, initial_guess, bounds):
        """Solve using Adaptive Genetic Algorithm.
        
        Args:
            initial_guess: Initial solution guess
            bounds: List of (min, max) bounds for each variable
            
        Returns:
            dict: Optimization results
        """
        try:
            logger.info("Starting Adaptive GA optimization...")
            start_time = time.time()
            
            # Get solver parameters from config
            aga_config = self.solver_config.get('solver_specific', {})
            pop_size = aga_config.get('initial_pop_size', 100)
            n_gen = aga_config.get('max_generations', 100)
            tournament_size = aga_config.get('tournament_size', 3)
            crossover_prob = aga_config.get('initial_crossover_rate', 0.9)
            mutation_prob = aga_config.get('initial_mutation_rate', 0.1)
            
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
            
            # Create adaptive callback with config parameters
            callback = AdaptiveCallback(
                solver=self,
                stagnation_limit=aga_config.get('stagnation_limit', 10),
                min_mutation_rate=aga_config.get('min_mutation_rate', 0.01),
                max_mutation_rate=aga_config.get('max_mutation_rate', 0.5),
                min_crossover_rate=aga_config.get('min_crossover_rate', 0.5),
                max_crossover_rate=aga_config.get('max_crossover_rate', 1.0),
                elite_size=aga_config.get('elite_size', 2),
                convergence_threshold=aga_config.get('convergence_threshold', 1e-6)
            )
            
            # Run optimization
            result = minimize(
                problem,
                algorithm,
                ('n_gen', n_gen),
                callback=callback,
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
                    message="Adaptive GA optimization failed to converge",
                    time=execution_time
                )
            
        except Exception as e:
            logger.error(f"Error in Adaptive GA solver: {str(e)}")
            return self.process_results(
                np.zeros_like(initial_guess),
                success=False,
                message=f"Error in Adaptive GA solver: {str(e)}"
            )
