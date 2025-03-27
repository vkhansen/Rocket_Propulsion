"""Base genetic algorithm solver implementation."""
import numpy as np
import time
from ...utils.config import logger
from .base_solver import BaseSolver
from ..objective import objective_with_penalty
from ..physics import calculate_stage_ratios, calculate_payload_fraction

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
        # Ensure bootstrap solution is properly initialized
        self.best_bootstrap_solution = None
        self.best_bootstrap_fitness = float('-inf')

    def initialize_population(self, other_solver_results=None):
        """Initialize population with random individuals and bootstrap solutions.
        
        Args:
            other_solver_results: Results from other solvers to use as bootstrap solutions
            
        Returns:
            Tuple of (population, fitness, feasibility, violations)
        """
        try:
            # Process bootstrap solutions if available
            bootstrap_solutions = []
            if other_solver_results is not None and len(other_solver_results) > 0:
                bootstrap_solutions = self.process_bootstrap_solutions(
                    other_solver_results,
                    n_perturbed=min(5, max(1, self.pop_size // 10)),  # Scale with population size
                    perturbation_scale=0.1
                )
                logger.info(f"Processed {len(bootstrap_solutions)} bootstrap solutions")
            
            # Generate random population for remaining slots
            remaining_size = max(0, self.pop_size - len(bootstrap_solutions))
            
            if remaining_size > 0:
                random_population = self._generate_random_population(remaining_size)
                
                if random_population is not None and len(random_population) > 0:
                    # Combine bootstrap solutions with random population
                    self.population = np.vstack([
                        np.array(bootstrap_solutions),
                        random_population
                    ]) if bootstrap_solutions else random_population
                else:
                    # Fallback to bootstrap solutions only if random generation failed
                    self.population = np.array(bootstrap_solutions) if bootstrap_solutions else None
            else:
                # Use only bootstrap solutions if they fill or exceed the population
                self.population = np.array(bootstrap_solutions[:self.pop_size])
            
            # Ensure we have a valid population
            if self.population is None or len(self.population) == 0:
                logger.warning("Failed to initialize population, using fallback method")
                self.population = self.initialize_population_lhs()
            
            # Ensure population size matches expected size
            if len(self.population) > self.pop_size:
                # Trim excess individuals
                self.population = self.population[:self.pop_size]
            elif len(self.population) < self.pop_size:
                # Add more random individuals if needed
                additional = self._generate_random_population(self.pop_size - len(self.population))
                if additional is not None and len(additional) > 0:
                    self.population = np.vstack([self.population, additional])
            
            # Initialize arrays for evaluation results
            fitness = np.zeros(self.pop_size)
            feasibility = np.zeros(self.pop_size, dtype=bool)
            violations = np.zeros(self.pop_size)
            
            # Evaluate initial population
            for i, individual in enumerate(self.population):
                # Ensure solution is feasible before evaluation
                self.population[i] = self.project_to_feasible(self.population[i])
                
                # Validate physics one more time
                is_valid, _ = self.validate_physics(self.population[i])
                if not is_valid:
                    # Replace with a balanced solution
                    first_stage_dv = 0.4 * self.TOTAL_DELTA_V
                    remaining_dv = self.TOTAL_DELTA_V - first_stage_dv
                    other_stages_dv = remaining_dv / (self.n_stages - 1) if self.n_stages > 1 else 0
                    
                    self.population[i, 0] = first_stage_dv
                    for k in range(1, self.n_stages):
                        self.population[i, k] = other_stages_dv
                
                fitness[i] = self.evaluate(self.population[i])
                feasibility[i], violations[i] = self.check_feasibility(self.population[i])
                
                # Update best solution if better
                self.update_best_solution(
                    self.population[i], 
                    fitness[i], 
                    feasibility[i], 
                    violations[i]
                )
            
            # Find and log the best initial solution
            best_idx = np.argmax(fitness)
            best_fitness = fitness[best_idx]
            best_is_feasible = feasibility[best_idx]
            best_violation = violations[best_idx]
            
            # Log initial population statistics
            feasible_count = np.sum(feasibility)
            logger.info(f"Initialized population with {len(self.population)} individuals ({feasible_count} feasible)")
            
            if best_is_feasible:
                logger.info(f"Best initial solution is feasible with fitness {best_fitness:.6f}")
            else:
                logger.info(f"Best initial solution is infeasible with fitness {best_fitness:.6f} and violation {best_violation:.6f}")
                
            # Make sure we have the best bootstrap solution properly tracked
            if self.best_bootstrap_solution is not None:
                bootstrap_fitness = self.evaluate(self.best_bootstrap_solution)
                bootstrap_is_feasible, bootstrap_violation = self.check_feasibility(self.best_bootstrap_solution)
                
                logger.info(f"Best bootstrap solution has fitness {bootstrap_fitness:.6f} (feasible: {bootstrap_is_feasible})")
                
                # If the best bootstrap solution is better than our current best, update it
                if bootstrap_is_feasible and (not best_is_feasible or bootstrap_fitness > best_fitness):
                    self.update_best_solution(
                        self.best_bootstrap_solution,
                        bootstrap_fitness,
                        bootstrap_is_feasible,
                        bootstrap_violation
                    )
                    logger.info(f"Updated best solution with better bootstrap solution")
            
            # Log population statistics
            self.print_population_stats(self.population, fitness)
            
            return self.population, fitness, feasibility, violations
            
        except Exception as e:
            logger.error(f"Error initializing population: {str(e)}")
            # Create a fallback population with safe defaults
            fallback_pop_size = max(10, self.pop_size // 2)  # Use a smaller size for fallback
            fallback_population = np.zeros((fallback_pop_size, self.n_stages))
            
            # Create a balanced solution for each individual
            for i in range(fallback_pop_size):
                # Distribute delta-v evenly with small variations
                base_dv = self.TOTAL_DELTA_V / self.n_stages
                for j in range(self.n_stages):
                    # Add small random variations (±10%)
                    fallback_population[i, j] = base_dv * np.random.uniform(0.9, 1.1)
                
                # Ensure constraint satisfaction
                fallback_population[i] = self.project_to_feasible(fallback_population[i])
            
            # Initialize arrays for evaluation results
            fallback_fitness = np.zeros(fallback_pop_size)
            fallback_feasibility = np.zeros(fallback_pop_size, dtype=bool)
            fallback_violations = np.zeros(fallback_pop_size)
            
            # Evaluate fallback population
            for i in range(fallback_pop_size):
                try:
                    fallback_fitness[i] = self.evaluate(fallback_population[i])
                    fallback_feasibility[i], fallback_violations[i] = self.check_feasibility(fallback_population[i])
                except Exception:
                    # If evaluation fails, use worst possible values
                    fallback_fitness[i] = float('inf')
                    fallback_feasibility[i] = False
                    fallback_violations[i] = float('inf')
            
            logger.warning(f"Using fallback population with {fallback_pop_size} individuals")
            self.population = fallback_population
            return fallback_population, fallback_fitness, fallback_feasibility, fallback_violations

    def _generate_random_population(self, size):
        """Generate a random population of given size.
        
        Args:
            size: Number of individuals to generate
            
        Returns:
            Random population
        """
        population = np.zeros((size, self.n_stages))
        
        for i in range(size):
            # Generate more balanced random solutions
            # Allocate delta-v more evenly across stages
            valid_solution = False
            max_attempts = 20  # Increase max attempts for finding valid solutions
            
            for attempt in range(max_attempts):
                # Approach 1: Generate with random weights
                weights = np.random.uniform(0.5, 1.5, self.n_stages)  # More balanced weights
                solution = weights * (self.TOTAL_DELTA_V / np.sum(weights))
                
                # Ensure minimum delta-v for each stage
                min_dv_threshold = 50.0  # 50 m/s minimum delta-v per stage
                if np.any(solution < min_dv_threshold):
                    # Redistribute from stages with more delta-v
                    for j in range(self.n_stages):
                        if solution[j] < min_dv_threshold:
                            deficit = min_dv_threshold - solution[j]
                            solution[j] = min_dv_threshold
                            
                            # Find stages with enough delta-v to donate
                            donor_stages = [s for s in range(self.n_stages) 
                                           if s != j and solution[s] > min_dv_threshold * 2]
                            
                            if donor_stages:
                                # Distribute deficit among donor stages
                                per_stage_reduction = deficit / len(donor_stages)
                                for donor in donor_stages:
                                    solution[donor] -= per_stage_reduction
                
                # Ensure the solution sums to TOTAL_DELTA_V
                solution = self.project_to_feasible(solution)
                
                # Validate physics
                is_valid, _ = self.validate_physics(solution)
                if is_valid:
                    population[i] = solution
                    valid_solution = True
                    break
            
            # If we couldn't generate a valid solution after max attempts,
            # use a balanced approach with staged delta-v distribution
            if not valid_solution:
                logger.warning(f"Could not generate valid physics for individual {i} after {max_attempts} attempts")
                
                # Create a balanced solution with decreasing delta-v per stage
                # First stage: 40-60%, second stage: 25-40%, third stage: remainder
                if self.n_stages == 3:
                    first_stage_pct = np.random.uniform(0.4, 0.6)
                    second_stage_pct = np.random.uniform(0.25, 0.4)
                    third_stage_pct = 1.0 - first_stage_pct - second_stage_pct
                    
                    # Ensure third stage gets at least 10%
                    if third_stage_pct < 0.1:
                        # Redistribute
                        deficit = 0.1 - third_stage_pct
                        third_stage_pct = 0.1
                        # Take from first and second proportionally
                        first_reduction = deficit * (first_stage_pct / (first_stage_pct + second_stage_pct))
                        second_reduction = deficit - first_reduction
                        first_stage_pct -= first_reduction
                        second_stage_pct -= second_reduction
                    
                    population[i,0] = first_stage_pct * self.TOTAL_DELTA_V
                    population[i,1] = second_stage_pct * self.TOTAL_DELTA_V
                    population[i,2] = third_stage_pct * self.TOTAL_DELTA_V
                else:
                    # For other numbers of stages, distribute evenly with small random variations
                    base_pct = 1.0 / self.n_stages
                    percentages = np.random.uniform(0.7, 1.3, self.n_stages) * base_pct
                    # Normalize to ensure sum is 1.0
                    percentages = percentages / np.sum(percentages)
                    
                    for j in range(self.n_stages):
                        population[i,j] = percentages[j] * self.TOTAL_DELTA_V
                
                # Final validation and adjustment
                population[i] = self.project_to_feasible(population[i])
                is_valid, _ = self.validate_physics(population[i])
                
                if not is_valid:
                    # Last resort: equal distribution with slight variations
                    equal_dv = self.TOTAL_DELTA_V / self.n_stages
                    for j in range(self.n_stages):
                        # Add small variations (±5%)
                        population[i,j] = equal_dv * np.random.uniform(0.95, 1.05)
                    
                    # Ensure constraint satisfaction
                    population[i] = self.project_to_feasible(population[i])
        
        return population

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
                is_feasible, _ = self.check_feasibility(candidate)
                if is_feasible:
                    feasible_candidates.append((idx, fitness_values[idx], _))
            
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
        
        # Physics-based validation: ensure minimum delta-v for each stage
        min_dv_threshold = 50.0  # 50 m/s minimum delta-v per stage
        for idx in range(self.n_stages):
            if child[idx] < min_dv_threshold:
                # Redistribute from other stages if this stage is too small
                deficit = min_dv_threshold - child[idx]
                child[idx] = min_dv_threshold
                
                # Find stages with enough delta-v to donate
                donor_stages = [s for s in range(self.n_stages) if s != idx and child[s] > min_dv_threshold * 2]
                if donor_stages:
                    # Distribute deficit among donor stages
                    per_stage_reduction = deficit / len(donor_stages)
                    for donor in donor_stages:
                        child[donor] -= per_stage_reduction
                else:
                    # If no suitable donors, try a different alpha value
                    return self.crossover(parent1, parent2)
        
        # Check for maximum delta-v percentage in any single stage
        max_percentage = 0.80  # No stage should have more than 80% of total delta-v
        stage_percentages = child / np.sum(child)
        if np.any(stage_percentages > max_percentage):
            # Find the stage with too much delta-v
            high_idx = np.argmax(stage_percentages)
            excess = (stage_percentages[high_idx] - max_percentage) * np.sum(child)
            
            # Reduce this stage to the maximum percentage
            child[high_idx] = max_percentage * np.sum(child)
            
            # Distribute excess to other stages proportionally
            other_indices = [idx for idx in range(self.n_stages) if idx != high_idx]
            if other_indices:
                per_stage_addition = excess / len(other_indices)
                for idx in other_indices:
                    child[idx] += per_stage_addition
        
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
        
        # Final physics validation
        try:
            # Check if the child solution produces valid physics
            stage_ratios, mass_ratios = calculate_stage_ratios(
                dv=child,
                G0=self.G0,
                ISP=self.ISP,
                EPSILON=self.EPSILON
            )
            
            # Verify no negative or invalid mass ratios
            if np.any(mass_ratios <= 0) or np.any(~np.isfinite(mass_ratios)):
                logger.debug(f"Crossover produced invalid mass ratios: {mass_ratios}")
                # Try again with a different alpha
                return self.crossover(parent1, parent2)
                
            # Verify payload fraction is positive
            payload_fraction = calculate_payload_fraction(mass_ratios)
            if payload_fraction <= 0 or not np.isfinite(payload_fraction):
                logger.debug(f"Crossover produced invalid payload fraction: {payload_fraction}")
                # Try again with a different alpha
                return self.crossover(parent1, parent2)
                
        except Exception as e:
            logger.debug(f"Physics validation failed during crossover: {str(e)}")
            # Try again with a different alpha
            return self.crossover(parent1, parent2)
        
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
        
        # Physics-based validation: ensure minimum delta-v for each stage
        min_dv_threshold = 50.0  # 50 m/s minimum delta-v per stage
        for idx in range(self.n_stages):
            if mutated[idx] < min_dv_threshold:
                # Redistribute from other stages if this stage is too small
                deficit = min_dv_threshold - mutated[idx]
                mutated[idx] = min_dv_threshold
                
                # Find stages with enough delta-v to donate
                donor_stages = [s for s in range(self.n_stages) if s != idx and mutated[s] > min_dv_threshold * 2]
                if donor_stages:
                    # Distribute deficit among donor stages
                    per_stage_reduction = deficit / len(donor_stages)
                    for donor in donor_stages:
                        mutated[donor] -= per_stage_reduction
                else:
                    # If no suitable donors, revert to original solution and try again with smaller mutation
                    return self.mutate(solution)
        
        # Check for maximum delta-v percentage in any single stage
        max_percentage = 0.80  # No stage should have more than 80% of total delta-v
        stage_percentages = mutated / np.sum(mutated)
        if np.any(stage_percentages > max_percentage):
            # Find the stage with too much delta-v
            high_idx = np.argmax(stage_percentages)
            excess = (stage_percentages[high_idx] - max_percentage) * np.sum(mutated)
            
            # Reduce this stage to the maximum percentage
            mutated[high_idx] = max_percentage * np.sum(mutated)
            
            # Distribute excess to other stages proportionally
            other_indices = [idx for idx in range(self.n_stages) if idx != high_idx]
            if other_indices:
                per_stage_addition = excess / len(other_indices)
                for idx in other_indices:
                    mutated[idx] += per_stage_addition
        
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
        
        # Final physics validation
        try:
            # Check if the mutated solution produces valid physics
            stage_ratios, mass_ratios = calculate_stage_ratios(
                dv=mutated,
                G0=self.G0,
                ISP=self.ISP,
                EPSILON=self.EPSILON
            )
            
            # Verify no negative or invalid mass ratios
            if np.any(mass_ratios <= 0) or np.any(~np.isfinite(mass_ratios)):
                logger.debug(f"Mutation produced invalid mass ratios: {mass_ratios}")
                # Try again with the original solution
                return self.mutate(solution)
                
            # Verify payload fraction is positive
            payload_fraction = calculate_payload_fraction(mass_ratios)
            if payload_fraction <= 0 or not np.isfinite(payload_fraction):
                logger.debug(f"Mutation produced invalid payload fraction: {payload_fraction}")
                # Try again with the original solution
                return self.mutate(solution)
                
        except Exception as e:
            logger.debug(f"Physics validation failed during mutation: {str(e)}")
            # Try again with the original solution
            return self.mutate(solution)
        
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
                best_current_fitness = fitness_values[best_feasible_idx]
                
                # Always check if the best bootstrap solution is better than our current best
                if self.best_bootstrap_solution is not None:
                    bootstrap_fitness = self.evaluate_solution(self.best_bootstrap_solution)
                    
                    # If bootstrap solution is better, use it instead
                    if bootstrap_fitness > best_current_fitness:
                        new_population[0] = self.best_bootstrap_solution.copy()
                        logger.info(f"Restored better bootstrap solution with fitness {bootstrap_fitness:.6f} (current best: {best_current_fitness:.6f})")
                    else:
                        # Otherwise use the current best
                        new_population[0] = population[best_feasible_idx].copy()
                        logger.debug(f"Preserved best feasible solution with fitness {best_current_fitness:.6f}")
                else:
                    # No bootstrap solution, use current best
                    new_population[0] = population[best_feasible_idx].copy()
                    logger.debug(f"Preserved best feasible solution with fitness {best_current_fitness:.6f}")
            else:
                # If no feasible solutions, preserve best overall
                best_idx = np.argmax(fitness_values)
                
                # Check if the best bootstrap solution is better
                if self.best_bootstrap_solution is not None:
                    bootstrap_fitness = self.evaluate_solution(self.best_bootstrap_solution)
                    
                    # If bootstrap is better, use it
                    if bootstrap_fitness > fitness_values[best_idx]:
                        new_population[0] = self.best_bootstrap_solution.copy()
                        logger.info(f"Restored better bootstrap solution with fitness {bootstrap_fitness:.6f}")
                    else:
                        # Otherwise use current best and project it
                        new_population[0] = population[best_idx].copy()
                        # Project this solution to make it feasible
                        new_population[0] = self.iterative_projection(new_population[0])
                        logger.debug(f"No feasible solutions found, preserving and projecting best overall")
                else:
                    # No bootstrap solution, use current best
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
            
            # Enhanced bootstrap solution preservation
            # Always verify if we have a bootstrap solution and explicitly ensure it's in the population
            if self.best_bootstrap_solution is not None:
                # First, ensure the bootstrap solution is properly tracked
                bootstrap_is_feasible, _ = self.check_feasibility(self.best_bootstrap_solution)
                if bootstrap_is_feasible:
                    bootstrap_fitness = self.evaluate_solution(self.best_bootstrap_solution)
                    
                    # Evaluate new population to identify the worst individual
                    new_fitness_values = self.evaluate_population(new_population)
                    
                    # Check if bootstrap solution is already in the new population
                    bootstrap_in_population = False
                    for i, individual in enumerate(new_population):
                        if np.array_equal(individual, self.best_bootstrap_solution):
                            bootstrap_in_population = True
                            logger.debug(f"Bootstrap solution already exists in population at position {i}")
                            break
                    
                    # If not already in population, replace the worst individual (except the elite at position 0)
                    if not bootstrap_in_population:
                        # Find worst individual starting from index 1 (preserve elite at index 0)
                        if len(new_population) > 1:
                            worst_indices = np.argsort(new_fitness_values)[:-1]  # Sort ascending, exclude the best
                            for worst_idx in worst_indices:
                                if worst_idx > 0:  # Skip the elite at position 0
                                    # Replace worst with bootstrap solution
                                    new_population[worst_idx] = self.best_bootstrap_solution.copy()
                                    logger.info(f"Replaced worst individual at position {worst_idx} with bootstrap solution (fitness: {bootstrap_fitness:.6f})")
                                    break
            
            # Randomly inject bootstrap solution one more time for diversity
            # This additional random placement helps prevent the solution from being lost
            if self.best_bootstrap_solution is not None and np.random.random() < 0.3:  # 30% chance
                if len(population) > 1:
                    random_idx = np.random.randint(1, len(population))
                    new_population[random_idx] = self.best_bootstrap_solution.copy()
                    logger.debug(f"Randomly injected bootstrap solution at position {random_idx} for diversity")
            
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

    def validate_physics(self, solution):
        """Validate that a solution produces physically realistic results.
        
        Args:
            solution: Delta-v vector to validate
            
        Returns:
            Tuple of (is_valid, payload_fraction)
        """
        try:
            # Check for NaN or inf values
            if not np.all(np.isfinite(solution)):
                return False, 0.0
                
            # Check for minimum delta-v threshold
            min_dv_threshold = 200.0  # 200 m/s minimum delta-v per stage
            if np.any(solution < min_dv_threshold):
                return False, 0.0
            
            # Check for maximum delta-v percentage in any single stage
            max_percentage = 0.80  # No stage should have more than 80% of total delta-v
            total_dv = np.sum(solution)
            stage_percentages = solution / total_dv
            if np.any(stage_percentages > max_percentage):
                logger.debug(f"Solution rejected: stage has {np.max(stage_percentages)*100:.1f}% of total delta-v (max allowed: {max_percentage*100:.1f}%)")
                return False, 0.0
                
            # Calculate stage ratios and mass ratios
            stage_ratios, mass_ratios = calculate_stage_ratios(
                dv=solution,
                G0=self.G0,
                ISP=self.ISP,
                EPSILON=self.EPSILON
            )
            
            # Check for valid mass ratios
            if np.any(mass_ratios <= 0) or np.any(~np.isfinite(mass_ratios)):
                return False, 0.0
                
            # Calculate payload fraction
            payload_fraction = calculate_payload_fraction(mass_ratios)
            
            # Check for valid payload fraction
            if payload_fraction <= 0 or not np.isfinite(payload_fraction):
                return False, 0.0
                
            # Check for unrealistically small payload fraction
            if payload_fraction < 1e-6:  # Less than 0.0001% payload is unrealistic
                return False, 0.0
                
            return True, payload_fraction
            
        except Exception as e:
            logger.debug(f"Physics validation error: {str(e)}")
            return False, 0.0

    def optimize(self):
        """Run genetic algorithm optimization."""
        try:
            # Initialize population
            self.population, self.fitness_values, _, _ = self.initialize_population()
            if self.population is None:
                raise ValueError("Failed to initialize population")
                
            # Main optimization loop
            for gen in range(self.n_gen):
                try:
                    # Update best solution
                    gen_best_idx = np.argmax(self.fitness_values)
                    gen_best_fitness = self.fitness_values[gen_best_idx]
                    gen_best_solution = self.population[gen_best_idx].copy()
                    
                    # Check feasibility
                    is_feasible, violation = self.check_feasibility(gen_best_solution)
                    
                    # Update best solution if better
                    if gen_best_fitness > self.best_fitness:
                        self.best_fitness = gen_best_fitness
                        self.best_solution = gen_best_solution.copy()
                        
                        # Also update the base solver's best solution
                        self.update_best_solution(gen_best_solution, gen_best_fitness, is_feasible, violation)
                    
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
            self.population, self.fitness_values, _, _ = self.initialize_population(other_solver_results)
            if self.population is None:
                raise ValueError("Failed to initialize population")
                
            # Main optimization loop
            for gen in range(self.n_gen):
                try:
                    # Update best solution
                    gen_best_idx = np.argmax(self.fitness_values)
                    gen_best_fitness = self.fitness_values[gen_best_idx]
                    gen_best_solution = self.population[gen_best_idx].copy()
                    
                    # Check feasibility
                    is_feasible, violation = self.check_feasibility(gen_best_solution)
                    
                    # Update best solution if better
                    if gen_best_fitness > self.best_fitness:
                        self.best_fitness = gen_best_fitness
                        self.best_solution = gen_best_solution.copy()
                        
                        # Also update the base solver's best solution
                        self.update_best_solution(gen_best_solution, gen_best_fitness, is_feasible, violation)
                    
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
