"""Base genetic algorithm solver implementation."""
import numpy as np
import time
from src.utils.config import logger
from src.optimization.solvers.base_solver import BaseSolver
from src.optimization.objective import objective_with_penalty
from src.optimization.physics import calculate_stage_ratios, calculate_payload_fraction

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
        """Initialize population with random solutions.
        
        Args:
            other_solver_results: Optional results from other solvers to bootstrap population
            
        Returns:
            Tuple of (population, fitness, feasibility, violations)
        """
        try:
            # Create initial population
            population = np.zeros((self.pop_size, self.n_stages))
            
            # Process bootstrap solutions using the base class method
            bootstrap_solutions = self.process_bootstrap_solutions(other_solver_results)
            logger.info(f"Using {len(bootstrap_solutions)} valid bootstrap solutions")
            
            # Validate bootstrap solutions for physics realism
            valid_bootstrap_solutions = []
            for solution in bootstrap_solutions:
                is_valid, payload_fraction = self.validate_physics(solution)
                if is_valid:
                    valid_bootstrap_solutions.append((solution, payload_fraction))
                else:
                    logger.warning(f"Rejected bootstrap solution with invalid physics: {solution}")
            
            # Sort valid bootstrap solutions by payload fraction (descending)
            valid_bootstrap_solutions.sort(key=lambda x: x[1], reverse=True)
            valid_bootstrap = [sol for sol, _ in valid_bootstrap_solutions]
            
            logger.info(f"After physics validation: {len(valid_bootstrap)} valid bootstrap solutions")
            
            # Determine how many solutions to use from bootstrap (up to 30% of population)
            bootstrap_count = min(len(valid_bootstrap), max(self.pop_size // 3, 5))
            
            # Add bootstrap solutions to population
            for i in range(min(bootstrap_count, len(valid_bootstrap))):
                if i == 0 and len(valid_bootstrap) > 0:
                    # Keep the best solution exactly as is
                    population[i] = valid_bootstrap[0].copy()
                    logger.info(f"Preserving best bootstrap solution exactly: {population[i]}")
                else:
                    # Add some noise to bootstrap solutions to increase diversity
                    idx = np.random.randint(0, len(valid_bootstrap))
                    solution = valid_bootstrap[idx].copy()
                    
                    # Add small noise (0.5% variation)
                    noise = np.random.uniform(-0.005, 0.005, self.n_stages)
                    solution = solution * (1 + noise)
                    
                    # Ensure the solution is valid (sums to TOTAL_DELTA_V)
                    solution = solution * (self.TOTAL_DELTA_V / np.sum(solution))
                    
                    # Validate physics after adding noise
                    is_valid, _ = self.validate_physics(solution)
                    if not is_valid:
                        # If invalid after noise, use original
                        solution = valid_bootstrap[idx].copy()
                    
                    population[i] = solution
                    logger.debug(f"Added bootstrap solution with noise to population[{i}]")
            
            # Generate remaining population using different methods for diversity
            remaining_count = self.pop_size - bootstrap_count
            if remaining_count > 0:
                logger.info(f"Generating {remaining_count} random solutions to complete population")
                
                # Use Latin Hypercube Sampling for 40% of remaining
                lhs_count = min(remaining_count, max(self.pop_size // 5, 10))
                if lhs_count > 0:
                    try:
                        lhs_population = self.initialize_population_lhs()[0:lhs_count]
                        
                        # Validate LHS solutions
                        for j in range(lhs_count):
                            is_valid, _ = self.validate_physics(lhs_population[j])
                            if not is_valid:
                                # Replace invalid solutions with a balanced one
                                first_stage_dv = 0.4 * self.TOTAL_DELTA_V
                                remaining_dv = self.TOTAL_DELTA_V - first_stage_dv
                                other_stages_dv = remaining_dv / (self.n_stages - 1) if self.n_stages > 1 else 0
                                
                                lhs_population[j, 0] = first_stage_dv
                                for k in range(1, self.n_stages):
                                    lhs_population[j, k] = other_stages_dv
                        
                        population[bootstrap_count:bootstrap_count+lhs_count] = lhs_population
                    except Exception as e:
                        logger.warning(f"Error using LHS sampling: {e}, falling back to uniform")
                        lhs_count = 0
                
                # Use uniform random for the rest
                uniform_count = remaining_count - lhs_count
                if uniform_count > 0:
                    uniform_population = self._generate_random_population(uniform_count)
                    population[bootstrap_count+lhs_count:] = uniform_population
            
            # Evaluate the population
            fitness = np.zeros(self.pop_size)
            feasibility = np.zeros(self.pop_size, dtype=bool)
            violations = np.zeros(self.pop_size)
            
            for i in range(self.pop_size):
                # Ensure solution is feasible before evaluation
                population[i] = self.project_to_feasible(population[i])
                
                # Validate physics one more time
                is_valid, _ = self.validate_physics(population[i])
                if not is_valid:
                    # Replace with a balanced solution
                    first_stage_dv = 0.4 * self.TOTAL_DELTA_V
                    remaining_dv = self.TOTAL_DELTA_V - first_stage_dv
                    other_stages_dv = remaining_dv / (self.n_stages - 1) if self.n_stages > 1 else 0
                    
                    population[i, 0] = first_stage_dv
                    for k in range(1, self.n_stages):
                        population[i, k] = other_stages_dv
                
                fitness[i] = self.evaluate(population[i])
                feasibility[i], violations[i] = self.check_feasibility(population[i])
                
                # Update best solution if better
                self.update_best_solution(
                    population[i], 
                    fitness[i], 
                    feasibility[i], 
                    violations[i]
                )
            
            # Log population statistics
            self.print_population_stats(population, fitness)
            
            return population, fitness, feasibility, violations
            
        except Exception as e:
            logger.error(f"Error initializing population: {str(e)}")
            # Fallback to completely random population
            population = self._generate_random_population(self.pop_size)
            fitness = np.zeros(self.pop_size)
            feasibility = np.zeros(self.pop_size, dtype=bool)
            violations = np.zeros(self.pop_size)
            
            # Evaluate the fallback population
            for i in range(self.pop_size):
                fitness[i] = self.evaluate(population[i])
                feasibility[i], violations[i] = self.check_feasibility(population[i])
                
                # Update best solution if better
                self.update_best_solution(
                    population[i], 
                    fitness[i], 
                    feasibility[i], 
                    violations[i]
                )
                
            return population, fitness, feasibility, violations

    def _generate_random_population(self, size):
        """Generate a random population of given size.
        
        Args:
            size: Number of individuals to generate
            
        Returns:
            Random population
        """
        population = np.zeros((size, self.n_stages))
        
        for i in range(size):
            # Generate random solution within bounds
            for j in range(self.n_stages):
                lower, upper = self.bounds[j]
                population[i,j] = np.random.uniform(lower, upper)
            
            # Ensure minimum delta-v for each stage
            min_dv_threshold = 50.0  # 50 m/s minimum delta-v per stage
            for j in range(self.n_stages):
                if population[i,j] < min_dv_threshold:
                    population[i,j] = min_dv_threshold
            
            # Project to feasible space
            population[i] = self.project_to_feasible(population[i])
            
            # Validate physics
            max_attempts = 10
            attempt = 0
            valid_physics = False
            
            while not valid_physics and attempt < max_attempts:
                try:
                    # Check if the solution produces valid physics
                    stage_ratios, mass_ratios = calculate_stage_ratios(
                        dv=population[i],
                        G0=self.G0,
                        ISP=self.ISP,
                        EPSILON=self.EPSILON
                    )
                    
                    # Verify no negative or invalid mass ratios
                    if np.any(mass_ratios <= 0) or np.any(~np.isfinite(mass_ratios)):
                        # Try again with a new random solution
                        for j in range(self.n_stages):
                            lower, upper = self.bounds[j]
                            population[i,j] = np.random.uniform(lower, upper)
                            if population[i,j] < min_dv_threshold:
                                population[i,j] = min_dv_threshold
                        population[i] = self.project_to_feasible(population[i])
                        attempt += 1
                        continue
                        
                    # Verify payload fraction is positive
                    payload_fraction = calculate_payload_fraction(mass_ratios)
                    if payload_fraction <= 0 or not np.isfinite(payload_fraction):
                        # Try again with a new random solution
                        for j in range(self.n_stages):
                            lower, upper = self.bounds[j]
                            population[i,j] = np.random.uniform(lower, upper)
                            if population[i,j] < min_dv_threshold:
                                population[i,j] = min_dv_threshold
                        population[i] = self.project_to_feasible(population[i])
                        attempt += 1
                        continue
                    
                    # If we get here, the solution is valid
                    valid_physics = True
                    
                except Exception as e:
                    # Try again with a new random solution
                    for j in range(self.n_stages):
                        lower, upper = self.bounds[j]
                        population[i,j] = np.random.uniform(lower, upper)
                        if population[i,j] < min_dv_threshold:
                            population[i,j] = min_dv_threshold
                    population[i] = self.project_to_feasible(population[i])
                    attempt += 1
            
            # If we couldn't generate a valid solution after max attempts,
            # use a simple approach with balanced delta-v distribution
            if not valid_physics:
                logger.warning(f"Could not generate valid physics for individual {i} after {max_attempts} attempts")
                # Create a balanced solution with first stage having 40% of delta-v and rest distributed evenly
                first_stage_dv = 0.4 * self.TOTAL_DELTA_V
                remaining_dv = self.TOTAL_DELTA_V - first_stage_dv
                other_stages_dv = remaining_dv / (self.n_stages - 1) if self.n_stages > 1 else 0
                
                population[i,0] = first_stage_dv
                for j in range(1, self.n_stages):
                    population[i,j] = other_stages_dv
            
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
                new_population[0] = population[best_feasible_idx].copy()
                best_current_fitness = fitness_values[best_feasible_idx]
                logger.debug(f"Preserved best feasible solution with fitness {best_current_fitness}")
                
                # Check if the best bootstrap solution is better than our current best
                if self.best_bootstrap_solution is not None:
                    bootstrap_fitness = self.evaluate_solution(self.best_bootstrap_solution)
                    if bootstrap_fitness > best_current_fitness:
                        new_population[0] = self.best_bootstrap_solution.copy()
                        logger.info(f"Restored better bootstrap solution with fitness {bootstrap_fitness}")
            else:
                # If no feasible solutions, preserve best overall
                best_idx = np.argmax(fitness_values)
                new_population[0] = population[best_idx].copy()
                
                # Check if the best bootstrap solution is better
                if self.best_bootstrap_solution is not None:
                    bootstrap_fitness = self.evaluate_solution(self.best_bootstrap_solution)
                    if bootstrap_fitness > fitness_values[best_idx]:
                        new_population[0] = self.best_bootstrap_solution.copy()
                        logger.info(f"Restored better bootstrap solution with fitness {bootstrap_fitness}")
                    else:
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
            min_dv_threshold = 50.0  # 50 m/s minimum delta-v per stage
            if np.any(solution < min_dv_threshold):
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
