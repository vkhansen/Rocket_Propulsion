"""Adaptive Genetic Algorithm Solver implementation."""
import numpy as np
from typing import Dict, List, Tuple
from src.utils.config import logger
from src.optimization.solvers.base_ga_solver import BaseGASolver

class AdaptiveGeneticAlgorithmSolver(BaseGASolver):
    """Adaptive Genetic Algorithm solver implementation."""
    
    def __init__(self, G0, ISP, EPSILON, TOTAL_DELTA_V, bounds, 
                 pop_size=100, n_gen=100, mutation_rate=0.1, crossover_rate=0.9, tournament_size=3):
        """Initialize the adaptive GA solver."""
        super().__init__(G0, ISP, EPSILON, TOTAL_DELTA_V, bounds, 
                        pop_size, n_gen, mutation_rate, crossover_rate, tournament_size)
        
        # Initialize adaptive parameters
        self.min_pop_size = 50
        self.max_pop_size = 300
        self.min_mutation_rate = 0.01
        self.max_mutation_rate = 0.4
        self.min_crossover_rate = 0.3
        self.max_crossover_rate = 1.0
        
        # Initialize tracking variables
        self.generations_without_improvement = 0
        self.diversity_history = []
        
    def update_parameters(self):
        """Update algorithm parameters based on progress."""
        try:
            if self.population is None or self.fitness_values is None:
                return
                
            # Calculate diversity
            diversity = self.calculate_diversity(self.population)
            self.diversity_history.append(diversity)
            
            # Check for improvement and update parameters
            if self.generations_without_improvement > 10:
                # Increase exploration
                self.mutation_rate = min(self.max_mutation_rate, self.mutation_rate * 1.5)
                self.pop_size = min(self.max_pop_size, int(self.pop_size * 1.2))
            else:
                # Reduce exploration
                self.mutation_rate = max(self.min_mutation_rate, self.mutation_rate * 0.9)
                self.pop_size = max(self.min_pop_size, int(self.pop_size * 0.9))
                
        except Exception as e:
            logger.error(f"Error updating parameters: {str(e)}")
            
    def optimize(self):
        """Run adaptive genetic algorithm optimization."""
        try:
            # Initialize population
            self.population = self.initialize_population()
            if self.population is None:
                raise ValueError("Failed to initialize population")
                
            # Main optimization loop
            for gen in range(self.n_gen):
                try:
                    # Evaluate population
                    self.fitness_values = self.evaluate_population(self.population)
                    if self.fitness_values is None:
                        raise ValueError("Failed to evaluate population")
                    
                    # Update best solution
                    gen_best_idx = np.argmax(self.fitness_values)
                    gen_best_fitness = self.fitness_values[gen_best_idx]
                    
                    if gen_best_fitness > self.best_fitness:
                        self.best_fitness = gen_best_fitness
                        self.best_solution = self.population[gen_best_idx].copy()
                        self.generations_without_improvement = 0
                    else:
                        self.generations_without_improvement += 1
                    
                    # Calculate statistics
                    avg_fitness = np.mean(self.fitness_values)
                    diversity = self.calculate_diversity(self.population)
                    improvement = ((gen_best_fitness - self.best_fitness) / abs(self.best_fitness)) * 100 if self.best_fitness != 0 else 0
                    
                    # Log progress
                    logger.info(f"Generation {gen + 1}/{self.n_gen}:")
                    logger.info(f"  Best Fitness: {gen_best_fitness:.6f}")
                    logger.info(f"  Avg Fitness: {avg_fitness:.6f}")
                    logger.info(f"  Population Diversity: {diversity:.6f}")
                    logger.info(f"  Improvement: {improvement:+.2f}%")
                    
                    # Update adaptive parameters
                    self.update_parameters()
                    
                    # Create next generation with new parameters
                    new_population = self.create_next_generation(self.population, self.fitness_values)
                    if new_population is None:
                        raise ValueError("Failed to create next generation")
                        
                    self.population = new_population
                    
                except Exception as e:
                    logger.error(f"Error in generation {gen + 1}: {str(e)}")
                    if gen == 0:  # If error in first generation, abort
                        raise
                    continue  # Otherwise try to continue with next generation
                    
            return self.best_solution, self.best_fitness
            
        except Exception as e:
            logger.error(f"Error in Adaptive GA optimization: {str(e)}")
            return None, None
