"""Adaptive Genetic Algorithm Solver implementation."""
import numpy as np
from typing import Dict, List, Tuple
from src.utils.config import logger
from src.optimization.solvers.base_ga_solver import BaseGASolver
import time

class AdaptiveGeneticAlgorithmSolver(BaseGASolver):
    """Adaptive Genetic Algorithm solver implementation."""
    
    def __init__(self, G0, ISP, EPSILON, TOTAL_DELTA_V, bounds, config=None,
                 pop_size=100, n_gen=100, mutation_rate=0.1, crossover_rate=0.9, tournament_size=3):
        """Initialize the adaptive GA solver."""
        # Get solver-specific parameters from config if provided
        if config is not None:
            solver_params = config.get('solver_specific', {})
            pop_size = solver_params.get('population_size', pop_size)
            n_gen = solver_params.get('n_generations', n_gen)
            mutation_rate = solver_params.get('initial_mutation_rate', mutation_rate)
            crossover_rate = solver_params.get('initial_crossover_rate', crossover_rate)
            tournament_size = solver_params.get('tournament_size', tournament_size)
        
        super().__init__(G0, ISP, EPSILON, TOTAL_DELTA_V, bounds, config,
                        pop_size=pop_size, n_gen=n_gen, mutation_rate=mutation_rate,
                        crossover_rate=crossover_rate, tournament_size=tournament_size)
        
        # Initialize adaptive parameters with config values if provided
        if config is not None:
            solver_params = config.get('solver_specific', {})
            self.min_pop_size = solver_params.get('min_population_size', 50)
            self.max_pop_size = solver_params.get('max_population_size', 300)
            self.min_mutation_rate = solver_params.get('min_mutation_rate', 0.01)
            self.max_mutation_rate = solver_params.get('max_mutation_rate', 0.4)
            self.min_crossover_rate = solver_params.get('min_crossover_rate', 0.3)
            self.max_crossover_rate = solver_params.get('max_crossover_rate', 1.0)
        else:
            # Default adaptive parameters
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
                    
                    # Calculate improvement with proper handling of edge cases
                    if np.isfinite(self.best_fitness) and np.isfinite(gen_best_fitness) and abs(self.best_fitness) > 1e-10:
                        improvement = ((gen_best_fitness - self.best_fitness) / abs(self.best_fitness)) * 100
                    else:
                        improvement = 0.0
                    
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

    def solve(self, initial_guess, bounds, other_solver_results=None):
        """Solve optimization problem.
        
        Args:
            initial_guess: Initial solution vector
            bounds: List of (min, max) bounds for each variable
            other_solver_results: Optional dictionary of solutions from other solvers
            
        Returns:
            Dictionary containing optimization results
        """
        try:
            logger.info(f"\nStarting {self.name} optimization...")
            logger.info(f"Population size: {self.pop_size}")
            logger.info(f"Number of generations: {self.n_gen}")
            logger.info(f"Initial mutation rate: {self.mutation_rate}")
            logger.info(f"Initial crossover rate: {self.crossover_rate}")
            logger.info(f"Tournament size: {self.tournament_size}")
            
            # Log if we're using other solver results
            if other_solver_results:
                logger.info(f"Using {len(other_solver_results)} solutions from other solvers as seeds")
            logger.info("=" * 50)
            
            start_time = time.time()
            
            # Run the adaptive GA optimization with other solver results
            self.population = self.initialize_population(other_solver_results)
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
                    
                    # Calculate improvement with proper handling of edge cases
                    if np.isfinite(self.best_fitness) and np.isfinite(gen_best_fitness) and abs(self.best_fitness) > 1e-10:
                        improvement = ((gen_best_fitness - self.best_fitness) / abs(self.best_fitness)) * 100
                    else:
                        improvement = 0.0
                    
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
                    
            # Post-optimization feasibility enforcement
            logger.info("Optimization completed, checking solution feasibility...")
            
            # Ensure we have a solution to work with
            if self.best_solution is None:
                logger.warning("No best solution found during optimization, using best from final population")
                if self.population is not None and len(self.population) > 0:
                    self.fitness_values = self.evaluate_population(self.population)
                    self.best_solution = self.population[np.argmax(self.fitness_values)].copy()
                else:
                    logger.error("No valid population available, using initial guess")
                    self.best_solution = initial_guess.copy()
            
            # Project the best solution to ensure it satisfies constraints
            logger.info(f"Original best solution: {self.best_solution}")
            projected_solution = self.iterative_projection(self.best_solution)
            logger.info(f"Projected solution: {projected_solution}")
            
            # Check if the projected solution is feasible
            is_feasible, violation = self.check_feasibility(projected_solution)
            logger.info(f"Projected solution feasibility: {is_feasible}, violation: {violation}")
            
            if is_feasible:
                logger.info("Projected solution is feasible, using it as final result")
                self.best_solution = projected_solution
                success = True
                message = "Optimization completed successfully with feasible solution"
            else:
                logger.warning(f"Projected solution still violates constraints: {violation}")
                
                # Try to find any feasible solution in the final population
                found_feasible = False
                for i, ind in enumerate(self.population):
                    # Project each solution
                    projected_ind = self.iterative_projection(ind)
                    is_feasible, violation = self.check_feasibility(projected_ind)
                    if is_feasible:
                        logger.info(f"Found feasible solution in population at index {i}")
                        self.best_solution = projected_ind.copy()
                        found_feasible = True
                        break
                
                if found_feasible:
                    success = True
                    message = "Found alternative feasible solution in population"
                else:
                    # If no feasible solution found, use the best projected solution anyway
                    logger.warning("No feasible solution found in population, using best projected solution")
                    self.best_solution = projected_solution
                    success = False
                    message = f"Solution violates constraints (violation={violation:.2e})"
            
            execution_time = time.time() - start_time
            
            return self.process_results(
                x=self.best_solution,
                success=success,
                message=message,
                n_iterations=self.n_gen,
                n_function_evals=self.n_gen * self.pop_size,
                time=execution_time,
                constraint_violation=violation if 'violation' in locals() else None
            )
        except Exception as e:
            logger.error(f"Error in AdaptiveGeneticAlgorithmSolver: {str(e)}")
            return self.process_results(
                x=initial_guess,
                success=False,
                message=str(e),
                n_iterations=0,
                n_function_evals=0,
                time=0.0
            )
