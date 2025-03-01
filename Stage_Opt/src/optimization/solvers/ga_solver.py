"""Genetic Algorithm solver implementation."""
from typing import List, Dict, Tuple, Optional
import time
from pymoo.optimize import minimize
from ...utils.config import logger
from .base_ga_solver import BaseGASolver
import numpy as np

class GeneticAlgorithmSolver(BaseGASolver):
    """Genetic Algorithm solver implementation using pymoo framework."""
    
    def __init__(self, config: Dict, problem_params: Dict):
        """Initialize GA solver with configuration and problem parameters.
        
        Args:
            config: Solver configuration dictionary
            problem_params: Problem-specific parameters
        """
        super().__init__(config, problem_params)
        self.solver_specific = self.solver_config.get('solver_specific', {})
        
        # Get solver parameters from config with validation
        self.n_generations = max(int(self.solver_specific.get('max_generations', 100)), 1)
        self.min_diversity = float(self.solver_specific.get('min_diversity', 1e-6))
        self.stagnation_generations = int(self.solver_specific.get('stagnation_generations', 10))
        self.stagnation_threshold = float(self.solver_specific.get('stagnation_threshold', 1e-6))
        
        # Initialize history tracking
        self.best_fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []
        self.diversity_history: List[float] = []
        
        logger.debug(
            f"Initialized {self.name} with parameters:\n"
            f"  max_generations={self.n_generations}\n"
            f"  min_diversity={self.min_diversity}\n"
            f"  stagnation_generations={self.stagnation_generations}\n"
            f"  stagnation_threshold={self.stagnation_threshold}"
        )

    def _log_generation_stats(self, algorithm) -> Tuple[float, float, float]:
        """Log statistics for current generation.
        
        Args:
            algorithm: Current state of the optimization algorithm
            
        Returns:
            Tuple of (best_fitness, avg_fitness, diversity)
        """
        try:
            pop = algorithm.pop
            fitness_values = np.array([ind.F[0] for ind in pop])
            best_fitness = float(np.min(fitness_values))
            avg_fitness = float(np.mean(fitness_values))
            diversity = float(np.std(fitness_values))
            
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            self.diversity_history.append(diversity)
            
            # Calculate improvement percentage
            if len(self.best_fitness_history) > 1:
                improvement = ((self.best_fitness_history[-2] - best_fitness) / 
                             abs(self.best_fitness_history[-2]) * 100)
            else:
                improvement = 0.0
                
            logger.info(f"Generation {algorithm.n_gen}/{self.n_generations}:")
            logger.info(f"  Best Fitness: {best_fitness:.6f}")
            logger.info(f"  Avg Fitness: {avg_fitness:.6f}")
            logger.info(f"  Population Diversity: {diversity:.6f}")
            logger.info(f"  Improvement: {improvement:+.2f}%")
            
            # Print convergence warning if diversity is too low
            if diversity < self.min_diversity:
                logger.warning("  Low population diversity detected - possible premature convergence")
                
            # Print stagnation warning if no improvement for many generations
            if len(self.best_fitness_history) > self.stagnation_generations:
                recent_improvement = abs(
                    (self.best_fitness_history[-self.stagnation_generations] - best_fitness) / 
                    self.best_fitness_history[-self.stagnation_generations]
                )
                if recent_improvement < self.stagnation_threshold:
                    logger.warning("  Optimization appears to be stagnating - consider adjusting parameters")
                    
            return best_fitness, avg_fitness, diversity
            
        except Exception as e:
            logger.error(f"Error logging generation stats: {e}")
            return float('inf'), float('inf'), 0.0

    def solve(self, initial_guess: np.ndarray, bounds: List[Tuple[float, float]]) -> Dict:
        """Solve using Genetic Algorithm.
        
        Args:
            initial_guess: Initial solution vector
            bounds: List of (min, max) tuples for each variable
            
        Returns:
            Dictionary containing optimization results
        """
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
            
            # Track execution time
            start_time = time.time()
            
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
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Process results
            success = res.success if hasattr(res, 'success') else True
            message = res.message if hasattr(res, 'message') else ""
            x = res.X if hasattr(res, 'X') else initial_guess
            n_gen = res.algorithm.n_gen if hasattr(res.algorithm, 'n_gen') else 0
            n_eval = res.algorithm.evaluator.n_eval if hasattr(res.algorithm, 'evaluator') else 0
            
            # Validate solution
            if not isinstance(x, np.ndarray) or not np.all(np.isfinite(x)):
                raise ValueError("Invalid solution found")
                
            # Log final statistics
            logger.info("\nOptimization completed:")
            logger.info(f"  Number of generations: {n_gen}")
            logger.info(f"  Number of evaluations: {n_eval}")
            if hasattr(res, 'F'):
                logger.info(f"  Final best fitness: {res.F[0]:.6f}")
            logger.info(f"  Success: {success}")
            if message:
                logger.info(f"  Message: {message}")
            logger.info(f"  Execution time: {execution_time:.2f} seconds")
            logger.info("=" * 50)
            
            # Get final statistics
            best_fitness = float('inf')
            if len(self.best_fitness_history) > 0:
                best_fitness = self.best_fitness_history[-1]
                
            # Add convergence info to message
            message = (
                f"{message}\n"
                f"Best fitness: {best_fitness:.6f}\n"
                f"Generations: {n_gen}/{self.n_generations}"
            )
            
            return self.process_results(
                x=x,
                success=success,
                message=message,
                n_iterations=n_gen,
                n_function_evals=n_eval,
                time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Error in GA solver: {str(e)}")
            return self.process_results(
                x=initial_guess,
                success=False,
                message=str(e),
                time=0.0
            )
