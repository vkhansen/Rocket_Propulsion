"""Genetic Algorithm solver implementation."""
from pymoo.optimize import minimize
from ...utils.config import logger
from .base_ga_solver import BaseGASolver
import numpy as np

class GeneticAlgorithmSolver(BaseGASolver):
    """Genetic Algorithm solver implementation."""
    
    def __init__(self, config, problem_params):
        """Initialize GA solver."""
        super().__init__(config, problem_params)
        self.solver_specific = self.solver_config.get('solver_specific', {})
        
        # Get solver parameters from config
        self.n_generations = int(self.solver_specific.get('max_generations', 100))
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.diversity_history = []
        
        logger.debug(f"Initialized {self.name} with parameters: "
                    f"max_generations={self.n_generations}")

    def _log_generation_stats(self, algorithm):
        """Log statistics for current generation."""
        pop = algorithm.pop
        fitness_values = np.array([ind.F[0] for ind in pop])
        best_fitness = np.min(fitness_values)
        avg_fitness = np.mean(fitness_values)
        diversity = np.std(fitness_values)
        
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)
        self.diversity_history.append(diversity)
        
        # Calculate improvement percentage
        if len(self.best_fitness_history) > 1:
            improvement = ((self.best_fitness_history[-2] - best_fitness) / 
                         self.best_fitness_history[-2] * 100)
        else:
            improvement = 0.0
            
        logger.info(f"Generation {algorithm.n_gen}/{self.n_generations}:")
        logger.info(f"  Best Fitness: {best_fitness:.6f}")
        logger.info(f"  Avg Fitness: {avg_fitness:.6f}")
        logger.info(f"  Population Diversity: {diversity:.6f}")
        logger.info(f"  Improvement: {improvement:+.2f}%")
        
        # Print convergence warning if diversity is too low
        if diversity < 1e-6:
            logger.warning("  Low population diversity detected - possible premature convergence")
            
        # Print stagnation warning if no improvement for many generations
        if len(self.best_fitness_history) > 10:
            recent_improvement = (self.best_fitness_history[-10] - best_fitness) / self.best_fitness_history[-10]
            if abs(recent_improvement) < 1e-6:
                logger.warning("  Optimization appears to be stagnating - consider adjusting parameters")

    def solve(self, initial_guess, bounds):
        """Solve using Genetic Algorithm."""
        try:
            logger.info(f"\nStarting {self.name} optimization...")
            logger.info(f"Population size: {self.pop_size}")
            logger.info(f"Number of generations: {self.n_generations}")
            logger.info(f"Mutation rate: {self.mutation_rate}")
            logger.info(f"Crossover rate: {self.crossover_rate}")
            logger.info(f"Tournament size: {self.tournament_size}")
            logger.info("=" * 50)
            
            # Setup problem and algorithm
            problem = self.create_problem(initial_guess, bounds)
            algorithm = self.create_algorithm()
            
            # Define callback for logging
            def callback(algorithm):
                self._log_generation_stats(algorithm)
            
            # Run optimization
            res = minimize(
                problem,
                algorithm,
                ('n_gen', self.n_generations),
                callback=callback,
                seed=1,
                verbose=False
            )
            
            # Process results
            success = res.success if hasattr(res, 'success') else True
            message = res.message if hasattr(res, 'message') else ""
            x = res.X if hasattr(res, 'X') else initial_guess
            n_gen = res.algorithm.n_gen if hasattr(res.algorithm, 'n_gen') else 0
            n_eval = res.algorithm.evaluator.n_eval if hasattr(res.algorithm, 'evaluator') else 0
            
            # Log final statistics
            logger.info("\nOptimization completed:")
            logger.info(f"  Number of generations: {n_gen}")
            logger.info(f"  Number of evaluations: {n_eval}")
            logger.info(f"  Final best fitness: {res.F[0]:.6f}")
            logger.info(f"  Success: {success}")
            if message:
                logger.info(f"  Message: {message}")
            logger.info("=" * 50)
            
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
