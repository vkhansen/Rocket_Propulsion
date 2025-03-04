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
        
        super().__init__(G0, ISP, EPSILON, TOTAL_DELTA_V, bounds, config,
                         pop_size=pop_size, n_gen=n_gen, mutation_rate=mutation_rate,
                         crossover_rate=crossover_rate, tournament_size=tournament_size)
        
        # Additional GA solver parameters
        self.n_generations = max(int(max_generations), 1)
        self.min_diversity = float(min_diversity)
        self.stagnation_generations = int(stagnation_generations)
        self.stagnation_threshold = float(stagnation_threshold)
        
        # GA specific parameters
        if config is not None:
            self.pop_size = int(config.get('ga', {}).get('population_size', 50))
            self.max_generations = int(config.get('ga', {}).get('max_generations', 100))
            self.crossover_rate = float(config.get('ga', {}).get('crossover_rate', 0.8))
            self.mutation_rate = float(config.get('ga', {}).get('mutation_rate', 0.2))
            self.mutation_strength = float(config.get('ga', {}).get('mutation_strength', 0.1))
            self.tournament_size = int(config.get('ga', {}).get('tournament_size', 3))
            self.elitism_rate = float(config.get('ga', {}).get('elitism_rate', 0.1))
            self.target_fitness = float(config.get('ga', {}).get('target_fitness', -1.0))
        else:
            self.pop_size = 50
            self.max_generations = 100
            self.crossover_rate = 0.8
            self.mutation_rate = 0.2
            self.mutation_strength = 0.1
            self.tournament_size = 3
            self.elitism_rate = 0.1
            self.target_fitness = -1.0
        
        # Tracking variables
        self.best_solution = None
        self.best_fitness = float('inf')
        self.best_is_feasible = False
        self.best_violation = float('inf')
        self.function_evaluations = 0
        
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

    def log_generation_stats(self, generation, population, fitness, feasibility):
        """Log statistics for current generation.
        
        Args:
            generation: Current generation number
            population: Current population
            fitness: Fitness values for population
            feasibility: Feasibility flags for population
            
        Returns:
            None
        """
        try:
            if population is None or len(population) == 0:
                logger.warning("Cannot log generation stats: population is empty")
                return
                
            # Calculate statistics
            valid_fitness = fitness[np.isfinite(fitness)]
            if len(valid_fitness) == 0:
                logger.warning("No valid fitness values to log")
                return
                
            best_idx = np.argmin(fitness)
            best_fitness = fitness[best_idx]
            avg_fitness = np.mean(valid_fitness)
            diversity = self.calculate_diversity(population)
            n_feasible = np.sum(feasibility)
            
            # Calculate improvement
            if len(self.best_fitness_history) > 0:
                prev_best = self.best_fitness_history[-1]
                if np.isfinite(prev_best) and np.isfinite(best_fitness):
                    improvement = (prev_best - best_fitness) / abs(prev_best) * 100 if prev_best != 0 else 0
                else:
                    improvement = 0
            else:
                improvement = 0
                
            # Store history
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            self.diversity_history.append(diversity)
            
            # Log statistics
            logger.info(f"Generation {generation}/{self.max_generations}:")
            logger.info(f"  Best Fitness: {best_fitness:.6f}")
            logger.info(f"  Avg Fitness: {avg_fitness:.6f}")
            logger.info(f"  Diversity: {diversity:.6f}")
            logger.info(f"  Feasible Solutions: {n_feasible}/{len(population)} ({n_feasible/len(population)*100:.1f}%)")
            logger.info(f"  Improvement: {improvement:+.2f}%")
            
            # Check for potential issues
            if diversity < self.min_diversity:
                logger.warning("  Low diversity detected - possible premature convergence")
                
            if len(self.best_fitness_history) > self.stagnation_generations:
                recent_history = self.best_fitness_history[-self.stagnation_generations:]
                if all(np.isfinite(recent_history)):
                    recent_improvement = abs((recent_history[0] - recent_history[-1]) / recent_history[0]) if recent_history[0] != 0 else 0
                    if recent_improvement < self.stagnation_threshold:
                        logger.warning("  Optimization appears to be stagnating")
            
        except Exception as e:
            logger.error(f"Error logging generation stats: {str(e)}")

    def calculate_diversity(self, population):
        """Calculate diversity of the population.
        
        Args:
            population: Current population
            
        Returns:
            Diversity measure (standard deviation of solutions)
        """
        try:
            if population is None or len(population) < 2:
                return 0.0
            
            # Calculate standard deviation across all dimensions
            std_devs = np.std(population, axis=0)
            
            # Normalize by the bounds range for each dimension
            bounds_ranges = np.array([upper - lower for lower, upper in self.bounds])
            normalized_std_devs = std_devs / bounds_ranges
            
            # Average across all dimensions
            diversity = float(np.mean(normalized_std_devs))
            
            return diversity
            
        except Exception as e:
            logger.error(f"Error calculating diversity: {str(e)}")
            return 0.0

    def tournament_selection(self, fitness, feasibility, violations):
        """Select parent using tournament selection.
        
        Args:
            fitness: Array of fitness values
            feasibility: Array of feasibility flags
            violations: Array of constraint violations
            
        Returns:
            Index of selected parent
        """
        # Select tournament participants
        tournament_size = min(self.tournament_size, len(fitness))
        participants = np.random.choice(len(fitness), tournament_size, replace=False)
        
        # First prioritize feasible solutions
        feasible_participants = [i for i in participants if feasibility[i]]
        
        if feasible_participants:
            # If there are feasible solutions, select the best one
            feasible_fitness = [fitness[i] for i in feasible_participants]
            best_idx = np.argmin(feasible_fitness)  # Minimization
            return feasible_participants[best_idx]
        else:
            # If no feasible solutions, select the one with lowest violation
            participant_violations = [violations[i] for i in participants]
            best_idx = np.argmin(participant_violations)
            return participants[best_idx]
            
    def crossover(self, parent1, parent2):
        """Perform crossover between two parents.
        
        Args:
            parent1: First parent solution
            parent2: Second parent solution
            
        Returns:
            Child solution
        """
        # Blend crossover (BLX-alpha)
        alpha = 0.3
        child = np.zeros_like(parent1)
        
        for i in range(len(parent1)):
            # Calculate range with alpha extension
            min_val = min(parent1[i], parent2[i])
            max_val = max(parent1[i], parent2[i])
            range_i = max_val - min_val
            
            # Extended range
            min_bound = min_val - alpha * range_i
            max_bound = max_val + alpha * range_i
            
            # Generate child value within extended range
            child[i] = np.random.uniform(min_bound, max_bound)
            
        return child
        
    def mutate(self, solution):
        """Mutate a solution.
        
        Args:
            solution: Solution to mutate
            
        Returns:
            Mutated solution
        """
        mutated = solution.copy()
        
        # Determine how many genes to mutate
        n_mutations = max(1, int(self.mutation_strength * len(solution)))
        
        # Select genes to mutate
        mutation_indices = np.random.choice(len(solution), n_mutations, replace=False)
        
        for idx in mutation_indices:
            # Get bounds for this gene
            lower, upper = self.bounds[idx]
            
            # Gaussian mutation
            sigma = (upper - lower) * 0.1  # 10% of range as standard deviation
            mutated[idx] += np.random.normal(0, sigma)
            
            # Ensure bounds
            mutated[idx] = np.clip(mutated[idx], lower, upper)
            
        return mutated

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
            start_time = time.time()
            # Set bounds if provided
            if bounds is not None:
                self.bounds = bounds
                
            # Initialize population
            logger.info(f"Initializing population of size {self.pop_size}")
            population, fitness, feasibility, violations = self.initialize_population(other_solver_results)
            
            # Initialize best solution tracking
            self.best_solution = None
            self.best_fitness = float('inf')
            self.best_is_feasible = False
            self.best_violation = float('inf')
            
            # Find initial best solution
            for i in range(len(population)):
                self.update_best_solution(
                    population[i],
                    fitness[i],
                    feasibility[i],
                    violations[i]
                )
            
            # Main GA loop
            generation = 0
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
            
            # Ensure we have a valid solution to return
            if self.best_solution is None:
                # If no feasible solution found, return the best infeasible one
                best_idx = np.argmin(fitness)
                self.best_solution = population[best_idx].copy()
                self.best_fitness = fitness[best_idx]
                self.best_is_feasible = feasibility[best_idx]
                self.best_violation = violations[best_idx]
                
                logger.warning(f"No feasible solution found. Returning best infeasible solution with fitness {self.best_fitness}")
            
            # Return results
            return self.process_results(
                x=self.best_solution,
                success=True,
                message='Optimization terminated successfully',
                n_iterations=min(generation + 1, self.max_generations),
                n_function_evals=self.function_evaluations,
                time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Error in GA solver: {str(e)}")
            return self.process_results(
                x=initial_guess if initial_guess is not None else np.zeros(len(self.bounds)),
                success=False,
                message=f'Error in GA solver: {str(e)}',
                n_iterations=0,
                n_function_evals=self.function_evaluations,
                time=0.0
            )

    def evaluate(self, solution):
        """Evaluate a single solution.
        
        Args:
            solution: Solution vector to evaluate
            
        Returns:
            Objective value
        """
        try:
            return self.evaluate_solution(solution)
        except Exception as e:
            logger.error(f"Error in evaluate: {str(e)}")
            return float('-inf')
