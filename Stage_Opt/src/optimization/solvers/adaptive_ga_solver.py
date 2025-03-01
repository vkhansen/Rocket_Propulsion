"""Adaptive Genetic Algorithm solver implementation."""
import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.mutation.pm import PM
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize
from ...utils.config import logger
from .base_solver import BaseSolver
from .pymoo_problem import RocketStageProblem

class AdaptiveGeneticAlgorithmSolver(BaseSolver):
    """Adaptive Genetic Algorithm solver implementation."""
    
    def __init__(self, config, problem_params):
        """Initialize Adaptive GA solver."""
        super().__init__(config, problem_params)
        self.solver_specific = self.solver_config.get('solver_specific', {})
        
        # Initialize adaptive parameters
        self.min_pop_size = int(self.solver_specific.get('min_pop_size', 50))
        self.max_pop_size = int(self.solver_specific.get('max_pop_size', 200))
        self.min_gen = int(self.solver_specific.get('min_gen', 50))
        self.max_gen = int(self.solver_specific.get('max_gen', 500))
        self.improvement_threshold = float(self.solver_specific.get('improvement_threshold', 0.01))
        self.adaptation_interval = int(self.solver_specific.get('adaptation_interval', 10))
        
        # Get solver parameters from config
        self.initial_pop_size = int(self.solver_specific.get('initial_pop_size', 100))
        self.max_generations = int(self.solver_specific.get('max_generations', 100))
        self.initial_mutation_rate = float(self.solver_specific.get('initial_mutation_rate', 0.1))
        self.initial_crossover_rate = float(self.solver_specific.get('initial_crossover_rate', 0.9))
        self.tournament_size = int(self.solver_specific.get('tournament_size', 3))
        
        logger.debug(f"Initialized {self.name} with parameters: "
                    f"min_pop_size={self.min_pop_size}, max_pop_size={self.max_pop_size}, "
                    f"min_gen={self.min_gen}, max_gen={self.max_gen}")
    
    def adapt_parameters(self, algorithm, history):
        """Adapt algorithm parameters based on performance."""
        if len(history) < 2:
            return
            
        # Get improvement in last interval
        current_best = history[-1]
        prev_best = history[-2]
        improvement = (prev_best - current_best) / prev_best
        
        # Adjust population size based on improvement
        if improvement < self.improvement_threshold:
            algorithm.pop_size = min(
                self.max_pop_size,
                int(algorithm.pop_size * 1.5)
            )
        else:
            algorithm.pop_size = max(
                self.min_pop_size,
                int(algorithm.pop_size * 0.8)
            )
            
        logger.debug(f"Adapted population size to {algorithm.pop_size}")
    
    def solve(self, initial_guess, bounds):
        """Solve using Adaptive Genetic Algorithm."""
        try:
            logger.info("Starting Adaptive GA optimization...")
            
            # Setup problem
            n_var = len(initial_guess)
            bounds = np.array(bounds)  # Convert bounds to numpy array
            problem = RocketStageProblem(
                solver=self,
                n_var=n_var,
                bounds=bounds
            )
            
            # Setup algorithm
            algorithm = GA(
                pop_size=self.initial_pop_size,
                sampling=FloatRandomSampling(),
                crossover=SBX(prob=self.initial_crossover_rate, eta=30),
                mutation=PM(prob=self.initial_mutation_rate, eta=30),
                selection=TournamentSelection(pressure=self.tournament_size),
                eliminate_duplicates=True
            )
            
            # Initialize history
            history = []
            
            def callback(algorithm):
                """Callback to track progress and adapt parameters."""
                if algorithm.n_gen > 0:
                    # Get minimum fitness value from current population
                    fitness_values = algorithm.pop.get("F")
                    history.append(np.min(fitness_values))
                    if len(history) >= 2 and algorithm.n_gen % self.adaptation_interval == 0:
                        self.adapt_parameters(algorithm, history)
            
            # Run optimization
            res = minimize(
                problem,
                algorithm,
                ('n_gen', self.max_generations),
                seed=1,
                callback=callback,
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
            logger.error(f"Error in Adaptive GA solver: {str(e)}")
            return self.process_results(
                x=initial_guess,
                success=False,
                message=str(e)
            )
