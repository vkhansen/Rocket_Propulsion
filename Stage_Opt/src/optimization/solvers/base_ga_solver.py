"""Base genetic algorithm solver implementation."""
import numpy as np
import time
from src.utils.config import logger
from src.optimization.solvers.base_solver import BaseSolver
from src.optimization.objective import objective_with_penalty

class BaseGASolver(BaseSolver):
    """Base genetic algorithm solver for stage optimization."""

    def __init__(self, G0, ISP, EPSILON, TOTAL_DELTA_V, bounds, pop_size=100, n_gen=100,
                 mutation_rate=0.1, crossover_rate=0.9, tournament_size=3):
        """Initialize solver with GA parameters."""
        super().__init__(G0, ISP, EPSILON, TOTAL_DELTA_V, bounds)
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.best_fitness = float('-inf')
        self.best_solution = None
        self.population = None
        self.fitness_values = None
        
    def initialize_population(self):
        """Initialize random population within bounds."""
        try:
            n_vars = len(self.bounds)
            population = np.zeros((self.pop_size, n_vars))
            for i in range(n_vars):
                lower, upper = self.bounds[i]
                population[:, i] = np.random.uniform(lower, upper, self.pop_size)
            return population
        except Exception as e:
            logger.error(f"Error initializing population: {str(e)}")
            return None

    def evaluate_population(self, population):
        """Evaluate fitness for entire population."""
        try:
            if population is None:
                return None
                
            fitness_values = np.zeros(len(population))
            for i, individual in enumerate(population):
                try:
                    fitness = objective_with_penalty(
                        dv=individual,
                        G0=self.G0,
                        ISP=self.ISP,
                        EPSILON=self.EPSILON,
                        TOTAL_DELTA_V=self.TOTAL_DELTA_V
                    )
                    fitness_values[i] = fitness if fitness is not None else float('-inf')
                except Exception as e:
                    logger.error(f"Error evaluating individual {i}: {str(e)}")
                    fitness_values[i] = float('-inf')
                    
            return fitness_values
            
        except Exception as e:
            logger.error(f"Error evaluating population: {str(e)}")
            return None

    def tournament_selection(self, population, fitness_values):
        """Select parent using tournament selection."""
        try:
            if population is None or fitness_values is None:
                return None
                
            tournament_indices = np.random.randint(0, len(population), self.tournament_size)
            tournament_fitness = fitness_values[tournament_indices]
            
            # Handle NaN or inf values
            valid_mask = np.isfinite(tournament_fitness)
            if not np.any(valid_mask):
                return population[np.random.choice(len(population))]
                
            winner_idx = tournament_indices[np.argmax(tournament_fitness[valid_mask])]
            return population[winner_idx].copy()
            
        except Exception as e:
            logger.error(f"Error in tournament selection: {str(e)}")
            if population is not None and len(population) > 0:
                return population[np.random.randint(0, len(population))].copy()
            return None

    def crossover(self, parent1, parent2):
        """Perform crossover between parents."""
        try:
            if parent1 is None or parent2 is None:
                return None, None
                
            if np.random.random() > self.crossover_rate:
                return parent1.copy(), parent2.copy()
                
            # Single point crossover
            point = np.random.randint(1, len(parent1))
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            
            return child1, child2
            
        except Exception as e:
            logger.error(f"Error in crossover: {str(e)}")
            return None, None

    def mutate(self, individual):
        """Perform mutation on individual."""
        try:
            if individual is None:
                return None
                
            mutated = individual.copy()
            for i in range(len(mutated)):
                if np.random.random() < self.mutation_rate:
                    lower, upper = self.bounds[i]
                    mutated[i] = np.random.uniform(lower, upper)
            return mutated
            
        except Exception as e:
            logger.error(f"Error in mutation: {str(e)}")
            return None

    def create_next_generation(self, population, fitness_values):
        """Create next generation through selection, crossover and mutation."""
        try:
            if population is None or fitness_values is None:
                return None
                
            new_population = np.zeros_like(population)
            
            # Elitism - preserve best individual
            best_idx = np.argmax(fitness_values)
            new_population[0] = population[best_idx].copy()
            
            # Create rest of new population
            for i in range(1, len(population), 2):
                try:
                    # Select parents
                    parent1 = self.tournament_selection(population, fitness_values)
                    parent2 = self.tournament_selection(population, fitness_values)
                    
                    if parent1 is None or parent2 is None:
                        # Use random selection as fallback
                        idx1, idx2 = np.random.choice(len(population), 2, replace=False)
                        parent1, parent2 = population[idx1].copy(), population[idx2].copy()
                    
                    # Crossover
                    child1, child2 = self.crossover(parent1, parent2)
                    
                    if child1 is None or child2 is None:
                        child1, child2 = parent1.copy(), parent2.copy()
                    
                    # Mutation
                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)
                    
                    if child1 is None:
                        child1 = parent1.copy()
                    if child2 is None:
                        child2 = parent2.copy()
                    
                    # Add to new population
                    if i < len(population):
                        new_population[i] = child1
                    if i + 1 < len(population):
                        new_population[i + 1] = child2
                        
                except Exception as e:
                    logger.error(f"Error creating individuals {i}/{i+1}: {str(e)}")
                    if i < len(population):
                        new_population[i] = population[i].copy()
                    if i + 1 < len(population):
                        new_population[i + 1] = population[i + 1].copy()
            
            return new_population
            
        except Exception as e:
            logger.error(f"Error creating next generation: {str(e)}")
            return None

    def calculate_diversity(self, population):
        """Calculate population diversity."""
        try:
            if population is None or len(population) < 2:
                return 0.0
                
            # Calculate mean and std of population
            pop_mean = np.mean(population, axis=0)
            pop_std = np.std(population, axis=0)
            
            # Normalize by bounds range
            bounds_range = np.array([upper - lower for lower, upper in self.bounds])
            normalized_std = np.mean(pop_std / bounds_range)
            
            return float(normalized_std)
            
        except Exception as e:
            logger.error(f"Error calculating diversity: {str(e)}")
            return 0.0

    def optimize(self):
        """Run genetic algorithm optimization."""
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
                    
                    # Create next generation
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
            logger.error(f"Error in GA solver: {str(e)}")
            return None, None

    def solve(self, initial_guess, bounds):
        """Run genetic algorithm optimization."""
        try:
            start_time = time.time()
            
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
                    
                    # Create next generation
                    new_population = self.create_next_generation(self.population, self.fitness_values)
                    if new_population is None:
                        raise ValueError("Failed to create next generation")
                        
                    self.population = new_population
                    
                except Exception as e:
                    logger.error(f"Error in generation {gen + 1}: {str(e)}")
                    if gen == 0:  # If error in first generation, abort
                        raise
                    continue  # Otherwise try to continue with next generation
                    
            execution_time = time.time() - start_time
            
            if self.best_solution is None:
                return self.process_results(
                    x=initial_guess,
                    success=False,
                    message="No valid solution found",
                    n_iterations=self.n_gen,
                    n_function_evals=self.n_gen * self.pop_size,
                    time=execution_time
                )
                
            return self.process_results(
                x=self.best_solution,
                success=True,
                message="Optimization completed successfully",
                n_iterations=self.n_gen,
                n_function_evals=self.n_gen * self.pop_size,
                time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Error in GA solver: {str(e)}")
            return self.process_results(
                x=initial_guess,
                success=False,
                message=str(e),
                n_iterations=0,
                n_function_evals=0,
                time=0.0
            )
