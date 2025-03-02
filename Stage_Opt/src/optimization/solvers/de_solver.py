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
        
    def mutation(self, population, target_idx, F):
        """Generate mutant vector using current-to-best/1 strategy."""
        current = population[target_idx].copy()
        
        # Select three random distinct vectors
        idxs = [i for i in range(self.population_size) if i != target_idx]
        a, b, c = population[np.random.choice(idxs, 3, replace=False)]
        
        # Generate mutant with adaptive step size
        mutant = current + F * (b - c)
        
        # Project to feasible space
        mutant = self.iterative_projection(mutant)
        
        return mutant
        
    def crossover(self, target, mutant):
        """Perform binomial crossover."""
        trial = np.copy(target)
        
        # Ensure at least one component changes
        j_rand = np.random.randint(self.n_stages)
        
        for j in range(self.n_stages):
            if j == j_rand or np.random.random() < self.crossover_prob:
                trial[j] = mutant[j]
                
        # Project to feasible space
        trial = self.iterative_projection(trial)
        
        return trial
        
    def solve(self, initial_guess, bounds):
        """Solve using Differential Evolution."""
        try:
            logger.info("Starting DE optimization...")
            start_time = time.time()
            
            # Initialize population using LHS
            population = self.initialize_population_lhs()
            
            # Evaluate initial population
            fitness = np.full(self.population_size, float('inf'), dtype=np.float64)
            for i in range(self.population_size):
                fitness[i] = self.evaluate_solution(population[i])[0]
                
            best_idx = np.argmin(fitness)
            
            stall_count = 0
            for generation in range(self.max_iterations):
                improved = False
                
                # Adaptive mutation rate
                F = self.mutation_min + (self.mutation_max - self.mutation_min) * np.random.random()
                
                for i in range(self.population_size):
                    # Generate trial vector
                    mutant = self.mutation(population, i, F)
                    trial = self.crossover(population[i], mutant)
                    
                    # Evaluate trial vector
                    is_feasible, violation = self.check_feasibility(trial)
                    trial_fitness = self.evaluate_solution(trial)[0]
                    
                    # Selection
                    if trial_fitness < fitness[i]:
                        population[i] = trial.copy()
                        fitness[i] = trial_fitness
                        
                        # Update best solution
                        if self.update_best_solution(trial, trial_fitness, is_feasible, violation):
                            improved = True
                            
                # Check convergence
                if not improved:
                    stall_count += 1
                    if stall_count >= self.stall_limit:
                        break
                else:
                    stall_count = 0
                    
            # Return best feasible solution found
            if self.best_feasible is not None:
                return self.best_feasible, self.best_feasible_score
            else:
                return None, float('inf')
                
        except Exception as e:
            logger.error(f"DE optimization failed: {str(e)}")
            return None, float('inf')
