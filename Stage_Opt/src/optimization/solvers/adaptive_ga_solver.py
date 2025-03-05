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
            self.max_projection_iterations = solver_params.get('max_projection_iterations', 100)
        else:
            # Default adaptive parameters
            self.min_pop_size = 50
            self.max_pop_size = 300
            self.min_mutation_rate = 0.01
            self.max_mutation_rate = 0.4
            self.min_crossover_rate = 0.3
            self.max_crossover_rate = 1.0
            self.max_projection_iterations = 100
        
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
            population_result = self.initialize_population(other_solver_results)
            if population_result is None:
                raise ValueError("Failed to initialize population")
                
            # Unpack the population result
            self.population, self.fitness_values, feasibility, violations = population_result
            
            # Store the best bootstrap solution before optimization begins
            original_bootstrap_solution = None
            original_bootstrap_fitness = float('-inf')
            original_bootstrap_payload = 0.0
            
            if self.best_bootstrap_solution is not None:
                original_bootstrap_solution = self.best_bootstrap_solution.copy()
                original_bootstrap_fitness = self.best_bootstrap_fitness
                original_bootstrap_payload = -original_bootstrap_fitness
                logger.info(f"Original bootstrap solution with payload fraction {original_bootstrap_payload:.6f} will be preserved")
            
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
                    
                    # Verify after each generation that we haven't degraded below the bootstrap solution
                    current_best_payload = -self.best_fitness if self.best_fitness != float('-inf') else 0
                    
                    # If we have a bootstrap solution and our current best is worse than it, restore the bootstrap
                    if original_bootstrap_solution is not None and current_best_payload < original_bootstrap_payload:
                        logger.warning(f"Current best payload fraction ({current_best_payload:.6f}) is worse than bootstrap ({original_bootstrap_payload:.6f}). Restoring bootstrap solution.")
                        # Restore the bootstrap solution as our best
                        self.best_solution = original_bootstrap_solution.copy()
                        self.best_fitness = original_bootstrap_fitness
                        self.best_is_feasible = True
                        
                    # Every 10 generations, log the current best payload fraction for monitoring
                    if gen % 10 == 0:
                        logger.info(f"Generation {gen} best payload fraction: {-self.best_fitness:.6f}")
                    
                except Exception as e:
                    logger.error(f"Error in generation {gen + 1}: {str(e)}")
                    if gen == 0:  # If error in first generation, abort
                        raise
                    continue  # Otherwise try to continue with next generation
                    
            # Post-optimization feasibility enforcement
            logger.info("Optimization completed, checking solution feasibility...")
            
            # Final check to ensure we haven't degraded below the bootstrap solution
            if original_bootstrap_solution is not None:
                current_best_payload = -self.best_fitness if self.best_fitness != float('-inf') else 0
                
                if current_best_payload < original_bootstrap_payload:
                    logger.warning(f"Final solution payload fraction ({current_best_payload:.6f}) is worse than bootstrap ({original_bootstrap_payload:.6f}). Restoring bootstrap solution.")
                    self.best_solution = original_bootstrap_solution.copy()
                    self.best_fitness = original_bootstrap_fitness
                    self.best_is_feasible = True
            
            # Ensure we have a solution to work with
            if self.best_solution is None:
                logger.warning("No best solution found during optimization, using best from final population")
                if self.population is not None and len(self.population) > 0:
                    self.fitness_values = self.evaluate_population(self.population)
                    best_idx = np.argmax(self.fitness_values)
                    self.best_solution = self.population[best_idx].copy()
                    self.best_fitness = self.fitness_values[best_idx]
                else:
                    logger.error("No valid population available, using initial guess")
                    self.best_solution = initial_guess.copy()
                    self.best_fitness = self.evaluate_solution(self.best_solution)
            
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
                        success = True  # Set success to True since we found a feasible solution
                        break
                
                if found_feasible:
                    message = "Found alternative feasible solution in population"
                else:
                    # If no feasible solution found, use the best projected solution anyway
                    logger.warning("No feasible solution found in population, using best projected solution")
                    self.best_solution = projected_solution
                    success = False
                    message = f"Solution violates constraints (violation={violation:.2e})"
            
            execution_time = time.time() - start_time
            
            # Final bootstrap solution comparison
            if original_bootstrap_solution is not None:
                final_best_payload = -self.best_fitness if self.best_fitness != float('-inf') else 0
                logger.info(f"Final solution payload fraction: {final_best_payload:.6f}")
                logger.info(f"Original bootstrap payload fraction: {original_bootstrap_payload:.6f}")
                logger.info(f"Improvement over bootstrap: {((final_best_payload - original_bootstrap_payload) / original_bootstrap_payload) * 100:.2f}%")
            
            return self.process_results(
                x=self.best_solution,
                success=success,  # This should be True if we found a feasible solution
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

    def iterative_projection(self, x: np.ndarray) -> np.ndarray:
        """Iteratively project solution to feasible space until convergence.
        
        Args:
            x: Solution vector
            
        Returns:
            Projected solution
        """
        try:
            # Ensure x is a 1D array
            x = np.asarray(x, dtype=np.float64).reshape(-1)
            
            # Initialize variables
            prev_x = x.copy()
            current_x = x.copy()
            max_iterations = self.max_projection_iterations
            converged = False
            
            # Iteratively apply projection
            for i in range(max_iterations):
                # Apply projection
                current_x = self.project_to_feasible(current_x)
                
                # Check for convergence
                if np.allclose(current_x, prev_x, rtol=1e-6, atol=1e-6):
                    converged = True
                    logger.debug(f"Projection converged after {i+1} iterations")
                    break
                    
                # Update previous solution
                prev_x = current_x.copy()
            
            if not converged:
                logger.warning(f"Projection did not converge after {max_iterations} iterations")
                
            return current_x
            
        except Exception as e:
            logger.error(f"Error in iterative projection: {str(e)}")
            # Fallback to simple projection
            return self.project_to_feasible(x)
