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

    def solve(self, initial_guess=None, bounds=None, other_solver_results=None):
        """Solve the optimization problem using genetic algorithm.
        
        Args:
            initial_guess: Initial solution vector (not used in GA)
            bounds: List of (min, max) bounds for each variable
            other_solver_results: Optional results from other solvers to bootstrap population
            
        Returns:
            Dictionary with optimization results
        """
        try:
            # Set bounds if provided
            if bounds is not None:
                self.bounds = bounds
                
            # Initialize population
            logger.info(f"Initializing population of size {self.pop_size}")
            population, fitness, feasibility, violations = self.initialize_population(other_solver_results)
            
            # Main GA loop
            for generation in range(self.max_generations):
                # Log progress
                if generation % 10 == 0:
                    self.log_generation_stats(generation, population, fitness, feasibility)
                
                # Check for early termination
                if self.best_fitness < self.target_fitness:
                    logger.info(f"Target fitness reached at generation {generation}")
                    break
                    
                # Create next generation
                new_population = np.zeros_like(population)
                new_fitness = np.zeros_like(fitness)
                new_feasibility = np.zeros_like(feasibility)
                new_violations = np.zeros_like(violations)
                
                # Elitism: keep best solutions
                elite_count = max(1, int(self.pop_size * self.elitism_rate))
                elite_indices = np.argsort(fitness)[:elite_count]
                
                for i, idx in enumerate(elite_indices):
                    new_population[i] = population[idx]
                    new_fitness[i] = fitness[idx]
                    new_feasibility[i] = feasibility[idx]
                    new_violations[i] = violations[idx]
                
                # Fill rest of population with offspring
                for i in range(elite_count, self.pop_size):
                    # Selection
                    parent1_idx = self.tournament_selection(fitness, feasibility, violations)
                    parent2_idx = self.tournament_selection(fitness, feasibility, violations)
                    
                    # Crossover
                    if np.random.random() < self.crossover_rate:
                        child = self.crossover(population[parent1_idx], population[parent2_idx])
                    else:
                        child = population[parent1_idx].copy()
                    
                    # Mutation
                    if np.random.random() < self.mutation_rate:
                        child = self.mutate(child)
                    
                    # Project to feasible space
                    child = self.project_to_feasible(child)
                    
                    # Evaluate
                    new_population[i] = child
                    new_fitness[i] = self.evaluate(child)
                    new_feasibility[i], new_violations[i] = self.check_feasibility(child)
                    
                    # Update best solution
                    self.update_best_solution(
                        child, 
                        new_fitness[i], 
                        new_feasibility[i], 
                        new_violations[i]
                    )
                
                # Update population
                population = new_population
                fitness = new_fitness
                feasibility = new_feasibility
                violations = new_violations
            
            # Final log
            self.log_generation_stats(self.max_generations, population, fitness, feasibility)
            
            # Return results
            return {
                'x': self.best_solution,
                'fun': self.best_fitness,
                'success': True,
                'message': 'Optimization terminated successfully',
                'nfev': self.function_evaluations,
                'nit': min(generation + 1, self.max_generations),
                'is_feasible': self.best_is_feasible,
                'violation': self.best_violation
            }
            
        except Exception as e:
            logger.error(f"Error in GA solver: {str(e)}")
            return {
                'x': None,
                'fun': float('inf'),
                'success': False,
                'message': f'Error in GA solver: {str(e)}',
                'nfev': self.function_evaluations,
                'nit': 0
            }
