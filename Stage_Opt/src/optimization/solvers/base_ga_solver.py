"""Base genetic algorithm solver implementation."""
import numpy as np
import time
from src.utils.config import logger
from src.optimization.solvers.base_solver import BaseSolver
from src.optimization.objective import objective_with_penalty

class BaseGASolver(BaseSolver):
    """Base genetic algorithm solver for stage optimization."""

    def __init__(self, G0, ISP, EPSILON, TOTAL_DELTA_V, bounds, config=None, pop_size=100, n_gen=100,
                 mutation_rate=0.1, crossover_rate=0.9, tournament_size=3, mutation_std=1.0):
        """Initialize solver with GA parameters."""
        super().__init__(G0, ISP, EPSILON, TOTAL_DELTA_V, bounds, config)
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.mutation_std = mutation_std
        self.best_fitness = float('-inf')
        self.best_solution = None
        self.population = None
        self.fitness_values = None
        self.n_stages = len(bounds)
        
    def initialize_population(self, other_solver_results=None):
        """Initialize population with solutions that satisfy constraints.
        
        Args:
            other_solver_results: Optional dictionary of solutions from other solvers
        
        Returns:
            List of initialized population solutions
        """
        population = []
        bootstrapped_count = 0
        rejected_count = 0
        
        # If other solver results are provided, add them first
        if other_solver_results is not None and isinstance(other_solver_results, dict):
            logger.info(f"Initializing population with solutions from {len(other_solver_results)} other solvers")
            for solver_name, result in other_solver_results.items():
                if 'x' in result and result.get('success', False):
                    solution = result['x']
                    
                    # Log the bootstrapped solution
                    logger.debug(f"Bootstrapped solution from {solver_name}: {solution}")
                    
                    # Ensure solution meets constraints
                    total = np.sum(solution)
                    if total > 0:
                        # Normalize to ensure total delta-v constraint is satisfied
                        solution = np.array(solution, dtype=np.float64)
                        solution *= self.TOTAL_DELTA_V / total
                        
                        # Ensure all values are within bounds
                        for i in range(self.n_stages):
                            lower, upper = self.bounds[i]
                            solution[i] = np.clip(solution[i], lower, upper)
                        
                        # Verify solution quality by evaluating objective
                        fitness, penalty, payload_fraction = objective_with_penalty(
                            dv=solution,
                            G0=self.G0,
                            ISP=self.ISP,
                            EPSILON=self.EPSILON,
                            TOTAL_DELTA_V=self.TOTAL_DELTA_V,
                            return_tuple=True
                        )
                        
                        # Only add solutions with finite fitness
                        if np.isfinite(fitness):
                            population.append(solution)
                            bootstrapped_count += 1
                            logger.info(f"Added solution from {solver_name} to GA initial population with fitness {fitness:.6f}, penalty {penalty:.6f}, payload fraction {payload_fraction:.6f}")
                        else:
                            rejected_count += 1
                            logger.warning(f"Rejected bootstrapped solution from {solver_name} with infinite fitness: {fitness}")
                    else:
                        rejected_count += 1
                        logger.warning(f"Rejected bootstrapped solution from {solver_name} with zero total delta-v")
        
        # Log how many bootstrapped solutions were added
        if bootstrapped_count > 0 or rejected_count > 0:
            logger.info(f"Bootstrapping summary: Added {bootstrapped_count} valid solutions, rejected {rejected_count} solutions")
        
        # Fill remaining population with random solutions
        remaining = self.pop_size - len(population)
        logger.info(f"Generating {remaining} random solutions to complete population")
        
        attempts = 0
        max_attempts = remaining * 10  # Allow multiple attempts to find valid solutions
        
        while len(population) < self.pop_size and attempts < max_attempts:
            attempts += 1
            
            # Generate random fractions that sum to 1
            fractions = np.random.random(self.n_stages)
            fractions /= np.sum(fractions)
            
            # Scale by total ΔV and ensure bounds
            solution = fractions * self.TOTAL_DELTA_V
            for i in range(self.n_stages):
                lower, upper = self.bounds[i]
                solution[i] = np.clip(solution[i], lower, upper)
            
            # Final normalization with high precision
            total = np.sum(solution)
            if total > 0:
                solution = np.array(solution, dtype=np.float64)  # Higher precision
                solution *= self.TOTAL_DELTA_V / total
                
                # Verify and adjust for exact constraint
                error = np.abs(np.sum(solution) - self.TOTAL_DELTA_V)
                if error > 1e-10:
                    # Distribute any remaining error proportionally
                    adjustment = (self.TOTAL_DELTA_V - np.sum(solution)) / self.n_stages
                    solution += adjustment
                
                # Verify solution quality for random solutions too
                if attempts % 10 == 0:  # Only check every 10 attempts to save computation
                    fitness, penalty, _ = objective_with_penalty(
                        dv=solution,
                        G0=self.G0,
                        ISP=self.ISP,
                        EPSILON=self.EPSILON,
                        TOTAL_DELTA_V=self.TOTAL_DELTA_V,
                        return_tuple=True
                    )
                    if not np.isfinite(fitness) or penalty > 1.0:  # Skip solutions with high penalties
                        continue  # Skip this solution if fitness is not finite or penalty is high
                
                population.append(solution)
        
        # If we couldn't generate enough valid solutions, duplicate existing ones
        while len(population) < self.pop_size:
            if len(population) > 0:
                # Duplicate a random solution with small perturbation
                idx = np.random.randint(0, len(population))
                solution = population[idx].copy()
                
                # Add small random perturbation
                perturbation = np.random.normal(0, 0.01, size=self.n_stages)
                solution += perturbation
                
                # Normalize to maintain total delta-v
                total = np.sum(solution)
                if total > 0:
                    solution *= self.TOTAL_DELTA_V / total
                
                # Ensure all values are within bounds
                for i in range(self.n_stages):
                    lower, upper = self.bounds[i]
                    solution[i] = np.clip(solution[i], lower, upper)
                
                population.append(solution)
            else:
                # If no valid solutions, create a uniform distribution as fallback
                solution = np.ones(self.n_stages) * self.TOTAL_DELTA_V / self.n_stages
                population.append(solution)
        
        # Log population statistics
        self.print_population_stats(population)
        
        return np.array(population)

    def print_population_stats(self, population, fitness_values=None):
        """Print statistics about the population."""
        try:
            if population is None or len(population) == 0:
                logger.info("Population is empty or None")
                return
                
            # Calculate basic statistics
            pop_mean = np.mean(population, axis=0)
            pop_std = np.std(population, axis=0)
            pop_min = np.min(population, axis=0)
            pop_max = np.max(population, axis=0)
            
            logger.info(f"Population statistics (size={len(population)}):")
            logger.info(f"  Mean: {pop_mean}")
            logger.info(f"  Std Dev: {pop_std}")
            logger.info(f"  Min: {pop_min}")
            logger.info(f"  Max: {pop_max}")
            
            # If fitness values are provided, print fitness statistics
            if fitness_values is not None and len(fitness_values) > 0:
                valid_fitness = fitness_values[np.isfinite(fitness_values)]
                if len(valid_fitness) > 0:
                    logger.info(f"Fitness statistics:")
                    logger.info(f"  Mean: {np.mean(valid_fitness)}")
                    logger.info(f"  Std Dev: {np.std(valid_fitness)}")
                    logger.info(f"  Min: {np.min(valid_fitness)}")
                    logger.info(f"  Max: {np.max(valid_fitness)}")
                    logger.info(f"  Valid fitness values: {len(valid_fitness)}/{len(fitness_values)}")
                else:
                    logger.warning("No valid fitness values in population")
            
        except Exception as e:
            logger.error(f"Error printing population statistics: {str(e)}")

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
        """Select parent using tournament selection with feasibility prioritization."""
        try:
            if population is None or fitness_values is None:
                return None
                
            tournament_indices = np.random.randint(0, len(population), self.tournament_size)
            tournament_fitness = fitness_values[tournament_indices]
            
            # Check feasibility of tournament candidates
            feasible_candidates = []
            for idx in tournament_indices:
                candidate = population[idx]
                is_feasible, violation = self.check_feasibility(candidate)
                if is_feasible:
                    feasible_candidates.append((idx, fitness_values[idx], violation))
            
            # If we have feasible candidates, select the best one
            if feasible_candidates:
                # Sort by fitness (higher is better)
                feasible_candidates.sort(key=lambda x: x[1], reverse=True)
                winner_idx = feasible_candidates[0][0]
                return population[winner_idx].copy()
            
            # Otherwise fall back to standard tournament selection
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
        """Perform crossover while maintaining total ΔV constraint."""
        # Perform crossover
        alpha = np.random.random()
        child = alpha * parent1 + (1 - alpha) * parent2
        
        # Ensure bounds constraints
        for i in range(self.n_stages):
            lower, upper = self.bounds[i]
            child[i] = np.clip(child[i], lower, upper)
        
        # High precision normalization
        total = np.sum(child)
        if total > 0:
            child = np.array(child, dtype=np.float64)  # Higher precision
            child *= self.TOTAL_DELTA_V / total
            
            # Verify and adjust for exact constraint
            error = np.abs(np.sum(child) - self.TOTAL_DELTA_V)
            if error > 1e-10:
                # Distribute any remaining error proportionally
                adjustment = (self.TOTAL_DELTA_V - np.sum(child)) / self.n_stages
                child += adjustment
        
        return child

    def mutate(self, solution):
        """Mutate a solution while maintaining total ΔV constraint."""
        mutated = solution.copy()
        
        # Select two random stages for mutation
        i, j = np.random.choice(self.n_stages, size=2, replace=False)
        
        # Generate random perturbation maintaining sum
        delta = np.random.normal(0, self.mutation_std)
        mutated[i] += delta
        mutated[j] -= delta
        
        # Ensure bounds constraints
        for idx in range(self.n_stages):
            lower, upper = self.bounds[idx]
            mutated[idx] = np.clip(mutated[idx], lower, upper)
        
        # High precision normalization
        total = np.sum(mutated)
        if total > 0:
            mutated = np.array(mutated, dtype=np.float64)  # Higher precision
            mutated *= self.TOTAL_DELTA_V / total
            
            # Verify and adjust for exact constraint
            error = np.abs(np.sum(mutated) - self.TOTAL_DELTA_V)
            if error > 1e-10:
                # Distribute any remaining error proportionally
                adjustment = (self.TOTAL_DELTA_V - np.sum(mutated)) / self.n_stages
                mutated += adjustment
        
        return mutated

    def create_next_generation(self, population, fitness_values):
        """Create next generation through selection, crossover and mutation with feasibility preservation."""
        try:
            if population is None or fitness_values is None:
                return None
                
            new_population = np.zeros_like(population)
            
            # Find feasible solutions in current population
            feasible_indices = []
            for i, individual in enumerate(population):
                is_feasible, _ = self.check_feasibility(individual)
                if is_feasible:
                    feasible_indices.append(i)
            
            # Elitism - preserve best feasible individual if available
            if feasible_indices:
                feasible_fitness = fitness_values[feasible_indices]
                best_feasible_idx = feasible_indices[np.argmax(feasible_fitness)]
                new_population[0] = population[best_feasible_idx].copy()
                logger.debug(f"Preserved best feasible solution with fitness {fitness_values[best_feasible_idx]}")
            else:
                # If no feasible solutions, preserve best overall
                best_idx = np.argmax(fitness_values)
                new_population[0] = population[best_idx].copy()
                
                # Project this solution to make it feasible
                new_population[0] = self.iterative_projection(new_population[0])
                logger.debug(f"No feasible solutions found, preserving and projecting best overall")
            
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
                    if np.random.random() < self.crossover_rate:
                        child1 = self.crossover(parent1, parent2)
                        child2 = self.crossover(parent2, parent1)
                    else:
                        child1 = parent1.copy()
                        child2 = parent2.copy()
                    
                    if child1 is None:
                        child1 = parent1.copy()
                    if child2 is None:
                        child2 = parent2.copy()
                    
                    # Mutation
                    if np.random.random() < self.mutation_rate:
                        child1 = self.mutate(child1)
                    if np.random.random() < self.mutation_rate:
                        child2 = self.mutate(child2)
                    
                    if child1 is None:
                        child1 = parent1.copy()
                    if child2 is None:
                        child2 = parent2.copy()
                    
                    # Project children to feasible space
                    child1 = self.iterative_projection(child1)
                    child2 = self.iterative_projection(child2)
                    
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
                    
                    # Print population statistics
                    self.print_population_stats(self.population, self.fitness_values)
                    
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

    def solve(self, initial_guess, bounds, other_solver_results=None):
        """Run genetic algorithm optimization.
        
        Args:
            initial_guess: Initial solution vector
            bounds: List of (min, max) tuples for each variable
            other_solver_results: Optional dictionary of solutions from other solvers
        
        Returns:
            Dictionary containing optimization results
        """
        try:
            start_time = time.time()
            
            # Initialize population with solutions from other solvers
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
                    
                    # Print population statistics
                    self.print_population_stats(self.population, self.fitness_values)
                    
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
