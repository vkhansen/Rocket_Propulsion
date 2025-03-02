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
                 population_size: int = 150, max_generations: int = 300,
                 mutation_min: float = 0.4, mutation_max: float = 0.9,
                 crossover_prob: float = 0.7):
        """Initialize DE solver with problem parameters and DE-specific settings."""
        super().__init__(G0, ISP, EPSILON, TOTAL_DELTA_V, bounds, config)
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_min = mutation_min
        self.mutation_max = mutation_max
        self.crossover_prob = crossover_prob
        self.n_stages = len(bounds)
        
    def initialize_population(self):
        """Initialize population using Latin Hypercube Sampling."""
        try:
            from scipy.stats import qmc
            
            # Use Latin Hypercube Sampling for better coverage
            sampler = qmc.LatinHypercube(d=self.n_stages)
            samples = sampler.random(n=self.population_size)
            
            population = np.zeros((self.population_size, self.n_stages))
            
            # Scale samples to stage-specific ranges
            for i in range(self.population_size):
                for j in range(self.n_stages):
                    lower, upper = self.bounds[j]
                    population[i,j] = lower + samples[i,j] * (upper - lower)
                    
                # Project to feasible space
                population[i] = self.iterative_projection(population[i])
                    
            return population
            
        except Exception as e:
            logger.warning(f"LHS initialization failed: {str(e)}, using uniform random")
            return self._uniform_random_init()
            
    def _uniform_random_init(self):
        """Fallback uniform random initialization."""
        population = np.zeros((self.population_size, self.n_stages))
        
        for i in range(self.population_size):
            # Generate random position within bounds
            for j in range(self.n_stages):
                lower, upper = self.bounds[j]
                population[i,j] = np.random.uniform(lower, upper)
                
            # Project to feasible space
            population[i] = self.iterative_projection(population[i])
            
        return population
        
    def mutation(self, population, target_idx, F):
        """Generate mutant vector using current-to-best/1 strategy."""
        current = population[target_idx]
        
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
            
            # Initialize population
            population = self.initialize_population()
            
            # Evaluate initial population
            fitness = np.array([self.evaluate_solution(ind)[0] for ind in population])
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx].copy()
            best_fitness = fitness[best_idx]
            
            # Track best feasible solution
            best_feasible = None
            best_feasible_fitness = float('inf')
            best_violation = float('inf')
            stall_count = 0
            
            for generation in range(self.max_generations):
                # Adaptive mutation rate
                F = self.mutation_min + (self.mutation_max - self.mutation_min) * np.random.random()
                
                improved = False
                for i in range(self.population_size):
                    # Generate trial vector
                    mutant = self.mutation(population, i, F)
                    trial = self.crossover(population[i], mutant)
                    
                    # Evaluate trial vector
                    is_feasible, violation = self.check_feasibility(trial)
                    trial_fitness = self.evaluate_solution(trial)[0]
                    
                    # Selection
                    if trial_fitness < fitness[i]:
                        if is_feasible or violation < best_violation:
                            population[i] = trial
                            fitness[i] = trial_fitness
                            improved = True
                            
                            # Update best solution
                            if is_feasible and trial_fitness < best_feasible_fitness:
                                best_feasible = trial.copy()
                                best_feasible_fitness = trial_fitness
                                stall_count = 0
                            elif not is_feasible and violation < best_violation:
                                best_violation = violation
                                
                # Check convergence
                if not improved:
                    stall_count += 1
                    if stall_count >= 30:  # No improvement in 30 generations
                        break
                        
            # Return best feasible solution found
            if best_feasible is not None:
                return {
                    'x': best_feasible,
                    'success': True,
                    'message': f"Found feasible solution after {generation} generations",
                    'n_iterations': generation,
                    'n_function_evals': generation * self.population_size
                }
            else:
                return {
                    'x': np.zeros(self.n_stages),
                    'success': False,
                    'message': f"No feasible solution found after {generation} generations",
                    'n_iterations': generation,
                    'n_function_evals': generation * self.population_size
                }
                
        except Exception as e:
            logger.error(f"DE optimization failed: {str(e)}")
            return {
                'x': np.zeros(self.n_stages),
                'success': False,
                'message': str(e),
                'n_iterations': 0,
                'n_function_evals': 0
            }
