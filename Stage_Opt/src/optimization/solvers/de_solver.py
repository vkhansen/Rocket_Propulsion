"""Differential Evolution solver implementation."""
import time
import numpy as np
from typing import Dict, List, Tuple
from ...utils.config import logger
from .base_solver import BaseSolver

class DifferentialEvolutionSolver(BaseSolver):
    """Differential Evolution solver implementation."""
    
    def __init__(self, G0: float, ISP: List[float], EPSILON: List[float], 
                 TOTAL_DELTA_V: float, bounds: List[Tuple[float, float]], config: Dict,
                 mutation_min: float = 0.4, mutation_max: float = 0.9,
                 crossover_prob: float = 0.7):
        """Initialize DE solver with problem parameters and DE-specific settings."""
        super().__init__(G0, ISP, EPSILON, TOTAL_DELTA_V, bounds, config)
        
        # DE-specific parameters
        self.mutation_min = float(mutation_min)
        self.mutation_max = float(mutation_max)
        self.crossover_prob = float(crossover_prob)
        
    def initialize_population(self):
        """Initialize population with balanced stage allocations."""
        population = np.zeros((self.population_size, self.n_stages), dtype=np.float64)
        alpha = np.ones(self.n_stages) * 5.0  # Concentration parameter for balanced distribution
        
        for i in range(self.population_size):
            # Generate balanced proportions using Dirichlet
            props = np.random.dirichlet(alpha)
            population[i] = props * self.TOTAL_DELTA_V
            
            # Enforce first stage constraints (15-80% of total)
            first_stage_min = 0.15 * self.TOTAL_DELTA_V
            first_stage_max = 0.80 * self.TOTAL_DELTA_V
            if population[i, 0] < first_stage_min:
                excess = first_stage_min - population[i, 0]
                population[i, 0] = first_stage_min
                # Redistribute excess proportionally to other stages
                remaining_props = population[i, 1:] / population[i, 1:].sum()
                population[i, 1:] -= excess * remaining_props
            elif population[i, 0] > first_stage_max:
                excess = population[i, 0] - first_stage_max
                population[i, 0] = first_stage_max
                # Redistribute excess proportionally to other stages
                remaining_props = population[i, 1:] / population[i, 1:].sum()
                population[i, 1:] += excess * remaining_props
            
            # Project to feasible space while preserving proportions
            population[i] = self.iterative_projection(population[i])
        
        return population

    def mutation(self, population, target_idx, F):
        """Generate mutant vector using current-to-best/1 strategy with improved balance."""
        current = population[target_idx].copy()
        
        # Select three random distinct vectors
        idxs = [i for i in range(self.population_size) if i != target_idx]
        a, b, c = population[np.random.choice(idxs, 3, replace=False)]
        
        # Adaptive mutation scaling based on stage position
        F_scaled = np.zeros(self.n_stages, dtype=np.float64)
        for j in range(self.n_stages):
            if j == 0:
                # Reduced mutation for first stage to maintain constraints
                F_scaled[j] = F * 0.7
            else:
                # Balanced mutation for other stages
                F_scaled[j] = F * 0.9
        
        # Generate mutant with stage-specific mutation
        mutant = current.copy()
        for j in range(self.n_stages):
            mutant[j] += F_scaled[j] * (b[j] - c[j])
        
        # Enforce stage balance constraints
        total_dv = np.sum(mutant)
        max_stage_dv = 0.8 * total_dv  # No stage should exceed 80% of total
        
        # Check and rebalance if any stage exceeds limit
        max_stage = np.max(mutant)
        if max_stage > max_stage_dv:
            excess = max_stage - max_stage_dv
            max_idx = np.argmax(mutant)
            mutant[max_idx] = max_stage_dv
            
            # Redistribute excess to other stages proportionally
            other_stages = list(range(self.n_stages))
            other_stages.remove(max_idx)
            props = mutant[other_stages] / np.sum(mutant[other_stages])
            mutant[other_stages] += excess * props
        
        # Project to feasible space
        mutant = self.iterative_projection(mutant)
        
        return mutant

    def crossover(self, target, mutant):
        """Perform binomial crossover with stage-specific rates."""
        trial = target.copy()
        
        # Stage-specific crossover probabilities
        cr_scaled = np.zeros(self.n_stages, dtype=np.float64)
        for j in range(self.n_stages):
            if j == 0:
                # Lower crossover rate for first stage
                cr_scaled[j] = self.crossover_prob * 0.8
            else:
                # Higher crossover rate for other stages
                cr_scaled[j] = self.crossover_prob * 1.1
        
        # Ensure at least one component is crossed
        j_rand = np.random.randint(self.n_stages)
        for j in range(self.n_stages):
            if j == j_rand or np.random.random() < cr_scaled[j]:
                trial[j] = mutant[j]
        
        return trial

    def solve(self, initial_guess, bounds):
        """Solve using Differential Evolution."""
        try:
            logger.info("Starting DE optimization...")
            start_time = time.time()
            
            # Initialize population using balanced distribution
            population = self.initialize_population()
            
            # Evaluate initial population
            fitness = np.full(self.population_size, float('inf'), dtype=np.float64)
            for i in range(self.population_size):
                fitness[i] = self.evaluate_solution(population[i])
            
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx].copy()
            best_fitness = fitness[best_idx]
            
            stall_count = 0
            for generation in range(self.max_iterations):
                improved = False
                
                # Adaptive mutation rate
                progress = generation / self.max_iterations
                F = self.mutation_min + (self.mutation_max - self.mutation_min) * (1 - progress)**2
                
                # Evolution loop
                for i in range(self.population_size):
                    # Generate trial vector
                    mutant = self.mutation(population, i, F)
                    trial = self.crossover(population[i], mutant)
                    
                    # Evaluate trial vector
                    trial_fitness = self.evaluate_solution(trial)
                    
                    # Selection
                    if trial_fitness <= fitness[i]:
                        population[i] = trial
                        fitness[i] = trial_fitness
                        
                        # Update best solution
                        if trial_fitness < best_fitness:
                            best_solution = trial.copy()
                            best_fitness = trial_fitness
                            improved = True
                            stall_count = 0
                
                if not improved:
                    stall_count += 1
                    if stall_count >= self.stall_limit:
                        logger.info(f"DE converged after {generation + 1} generations")
                        break
                
                # Log progress periodically
                if (generation + 1) % 10 == 0:
                    logger.info(f"DE generation {generation + 1}/{self.max_iterations}, "
                              f"best fitness: {best_fitness:.6f}")
            
            execution_time = time.time() - start_time
            return self.process_results(
                best_solution,
                success=True,
                message="DE optimization completed successfully",
                n_iterations=generation + 1,
                n_function_evals=(generation + 1) * self.population_size,
                time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Error in DE optimization: {str(e)}")
            return self.process_results(
                initial_guess,
                success=False,
                message=f"DE optimization failed: {str(e)}",
                time=time.time() - start_time
            )
