import time
from typing import List, Tuple
import numpy as np
from pymoo.optimize import minimize
from ...utils.config import logger
from .base_ga_solver import BaseGASolver

class GeneticAlgorithmSolver(BaseGASolver):
    """Genetic Algorithm solver implementation using pymoo framework."""
    
    def __init__(self, G0, ISP, EPSILON, TOTAL_DELTA_V, bounds, config=None, pop_size=100, n_gen=100, 
                 mutation_rate=0.1, crossover_rate=0.9, tournament_size=3,
                 max_generations=100, min_diversity=1e-6, stagnation_generations=10, stagnation_threshold=1e-6):
        """Initialize GA solver with direct problem parameters and GA settings.

        Args:
            G0: Gravitational constant
            ISP: List of specific impulse values for each stage
            EPSILON: List of structural coefficients for each stage
            TOTAL_DELTA_V: Required total delta-v
            bounds: List of (min, max) bounds for each variable
            config: Optional solver configuration dictionary
            pop_size: Population size
            n_gen: Number of generations
            mutation_rate: Mutation rate
            crossover_rate: Crossover rate
            tournament_size: Tournament size for selection
            max_generations: Maximum generations for the underlying pymoo algorithm
            min_diversity: Minimum diversity threshold
            stagnation_generations: Number of generations to consider stagnation
            stagnation_threshold: Threshold for stagnation detection
        """
        # Get solver-specific parameters from config if provided
        if config is not None:
            solver_params = config.get('solver_specific', {})
            pop_size = solver_params.get('population_size', pop_size)
            n_gen = solver_params.get('n_generations', n_gen)
            mutation_rate = solver_params.get('mutation_rate', mutation_rate)
            crossover_rate = solver_params.get('crossover_rate', crossover_rate)
            tournament_size = solver_params.get('tournament_size', tournament_size)
            max_generations = solver_params.get('max_generations', max_generations)
            min_diversity = solver_params.get('min_diversity', min_diversity)
            stagnation_generations = solver_params.get('stagnation_generations', stagnation_generations)
            stagnation_threshold = solver_params.get('stagnation_threshold', stagnation_threshold)
        
        super().__init__(G0, ISP, EPSILON, TOTAL_DELTA_V, bounds,
                         pop_size=pop_size, n_gen=n_gen, mutation_rate=mutation_rate,
                         crossover_rate=crossover_rate, tournament_size=tournament_size)
        
        # Additional GA solver parameters
        self.n_generations = max(int(max_generations), 1)
        self.min_diversity = float(min_diversity)
        self.stagnation_generations = int(stagnation_generations)
        self.stagnation_threshold = float(stagnation_threshold)
        
        # Initialize history tracking
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.diversity_history = []
        
        logger.debug(
            f"Initialized {self.name} with parameters:\n"
            f"  pop_size={pop_size}, n_gen={n_gen}, mutation_rate={mutation_rate}, crossover_rate={crossover_rate}, tournament_size={tournament_size}\n"
            f"  max_generations={self.n_generations}, min_diversity={self.min_diversity}, "
            f"stagnation_generations={self.stagnation_generations}, stagnation_threshold={self.stagnation_threshold}"
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

    def solve(self, initial_guess: np.ndarray, bounds: List[Tuple[float, float]]) -> dict:
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
            
            # Setup problem and algorithm here (using pymoo if needed)
            # For demonstration, using the base GA solver routine
            start_time = time.time()
            
            # Initialize population
            self.population = self.initialize_population()
            if self.population is None:
                raise ValueError("Failed to initialize population")
            
            # Main optimization loop
            for gen in range(self.n_gen):
                self.fitness_values = self.evaluate_population(self.population)
                if self.fitness_values is None:
                    raise ValueError("Failed to evaluate population")
                
                gen_best_idx = np.argmax(self.fitness_values)
                gen_best_fitness = self.fitness_values[gen_best_idx]
                
                if gen_best_fitness > self.best_fitness:
                    self.best_fitness = gen_best_fitness
                    self.best_solution = self.population[gen_best_idx].copy()
                
                avg_fitness = np.mean(self.fitness_values)
                diversity = self.calculate_diversity(self.population)
                
                self.best_fitness_history.append(gen_best_fitness)
                self.avg_fitness_history.append(avg_fitness)
                self.diversity_history.append(diversity)
                
                logger.info(f"Generation {gen + 1}/{self.n_gen} - Best: {gen_best_fitness:.6f}, Avg: {avg_fitness:.6f}, Diversity: {diversity:.6f}")
                
                new_population = self.create_next_generation(self.population, self.fitness_values)
                if new_population is None:
                    raise ValueError("Failed to create next generation")
                
                self.population = new_population
                
            execution_time = time.time() - start_time
            
            return self.process_results(
                x=self.best_solution if self.best_solution is not None else initial_guess,
                success=True,
                message="Optimization completed successfully",
                n_iterations=self.n_gen,
                n_function_evals=self.n_gen * self.pop_size,
                time=execution_time
            )
        except Exception as e:
            logger.error(f"Error in GeneticAlgorithmSolver: {str(e)}")
            return self.process_results(
                x=initial_guess,
                success=False,
                message=str(e),
                n_iterations=0,
                n_function_evals=0,
                time=0.0
            )
