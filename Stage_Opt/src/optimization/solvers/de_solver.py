"""Differential Evolution solver implementation with enhanced constraint handling."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from ...utils.config import logger
from .base_solver import BaseSolver
from ..objective import objective_with_penalty
from scipy.stats import qmc

class DifferentialEvolutionSolver(BaseSolver):
    """Differential Evolution solver implementation with enhanced constraint handling."""
    
    def __init__(self, G0: float, ISP: list[float], EPSILON: list[float],
                 TOTAL_DELTA_V: float, bounds: list[tuple[float, float]], config: dict):
        """Initialize DE solver with problem parameters."""
        super().__init__(G0, ISP, EPSILON, TOTAL_DELTA_V, bounds, config)
        
        # DE-specific parameters with tuned values
        self.population_size = 150  # Increased for better coverage
        self.mutation_min = 0.4  # Higher mutation for exploration
        self.mutation_max = 0.9  # Upper bound for mutation
        self.recombination = 0.95  # Higher recombination for better mixing
        self.strategy = 'best1bin'  # More exploitative strategy
        self.max_iterations = 3000  # Increased iterations for convergence
        self.tol = 1e-8  # Stricter tolerance
        self.atol = 1e-8  # Absolute tolerance
        self.update_nlast = 30  # Check last N generations for convergence
        self.stall_generations = 100  # Increased stall limit
        
        # Adaptive parameters
        self.adaptive_penalty = True
        self.penalty_factor = 10.0
        self.penalty_growth = 5.0  # More aggressive penalty growth
        self.feasibility_threshold = 1e-8  # Tighter feasibility check
        
    def iterative_projection(self, x, max_iter=10, tol=1e-10):
        """Iteratively project solution until constraints are satisfied."""
        x_proj = x.copy()
        for _ in range(max_iter):
            # First ensure bounds constraints
            for i in range(self.n_stages):
                lower, upper = self.bounds[i]
                x_proj[i] = np.clip(x_proj[i], lower, upper)
            
            # Check total ΔV constraint
            total = np.sum(x_proj)
            error = abs(total - self.TOTAL_DELTA_V)
            
            if error <= tol:
                break
                
            # Scale to match total ΔV
            x_proj *= self.TOTAL_DELTA_V / total
            
            # Re-check bounds after scaling
            for i in range(self.n_stages):
                lower, upper = self.bounds[i]
                x_proj[i] = np.clip(x_proj[i], lower, upper)
                
            # Distribute any remaining error proportionally
            remaining = self.TOTAL_DELTA_V - np.sum(x_proj)
            if abs(remaining) > tol:
                adjustment = remaining / self.n_stages
                x_proj += adjustment
                
        return x_proj

    def evaluate_population(self, population):
        """Evaluate population with adaptive penalties."""
        scores = np.zeros(len(population))
        feasible_count = 0
        
        for i, x in enumerate(population):
            try:
                # Get objective and constraint components
                obj, dv_const, phys_const = self.evaluate_solution(x, return_components=True)
                
                # Apply adaptive penalty
                total_violation = dv_const + phys_const
                
                if total_violation <= self.feasibility_threshold:
                    feasible_count += 1
                    scores[i] = obj
                else:
                    if self.adaptive_penalty:
                        # Exponential penalty scaling
                        penalty = self.penalty_factor * np.exp(total_violation)
                        if total_violation > 0.1:
                            penalty *= self.penalty_growth
                        scores[i] = obj + penalty * total_violation
                    else:
                        scores[i] = obj + self.penalty_factor * total_violation
                    
            except Exception as e:
                logger.error(f"Error evaluating solution: {str(e)}")
                scores[i] = float('inf')
        
        if feasible_count == 0:
            logger.warning(f"No feasible solutions in population")
                
        return scores

    def initialize_population(self):
        """Initialize population using Latin Hypercube Sampling."""
        try:
            # Create Latin Hypercube sampler
            sampler = qmc.LatinHypercube(d=self.n_stages)
            
            # Generate samples in [0,1] space
            samples = sampler.random(n=self.population_size)
            
            # Scale to ensure total ΔV constraint
            population = np.zeros((self.population_size, self.n_stages))
            scale_factor = self.TOTAL_DELTA_V / self.n_stages
            
            for i in range(self.population_size):
                # Initial distribution proportional to total ΔV
                population[i] = samples[i] * scale_factor
                
                # Ensure sum equals total ΔV and constraints are satisfied
                population[i] = self.iterative_projection(population[i])
                
            return population
            
        except Exception as e:
            logger.warning(f"LHS initialization failed: {str(e)}, using uniform random")
            return self._uniform_random_init()

    def _uniform_random_init(self):
        """Fallback uniform random initialization."""
        population = np.zeros((self.population_size, self.n_stages))
        scale_factor = self.TOTAL_DELTA_V / self.n_stages
        
        for i in range(self.population_size):
            # Initialize around equal distribution
            population[i] = np.random.normal(scale_factor, scale_factor * 0.1, self.n_stages)
            population[i] = self.iterative_projection(population[i])
            
        return population

    def optimize(self):
        """Run differential evolution optimization."""
        try:
            # Initialize population
            population = self.initialize_population()
            scores = self.evaluate_population(population)
            
            # Track best solution
            best_idx = np.argmin(scores)
            best_score = scores[best_idx]
            best_solution = population[best_idx].copy()
            best_feasible_score = float('inf')
            best_feasible_solution = None
            
            # Main optimization loop
            iteration = 0
            stall_count = 0
            history = []
            
            while iteration < self.max_iterations:
                # Store previous best for convergence check
                prev_best = best_score
                improved = False
                
                # Evolve population
                for i in range(self.population_size):
                    # Select parents
                    idxs = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                    
                    # Adaptive mutation rate
                    progress = iteration / self.max_iterations
                    mutation = self.mutation_max - progress * (self.mutation_max - self.mutation_min)
                    
                    # Create trial vector
                    if self.strategy == 'best1bin':
                        mutant = population[best_idx] + mutation * (b - c)
                    else:
                        mutant = a + mutation * (b - c)
                    
                    # Crossover with higher probability
                    cross_points = np.random.rand(self.n_stages) < self.recombination
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.n_stages)] = True
                    trial = np.where(cross_points, mutant, population[i])
                    
                    # Iterative projection to feasible space
                    trial = self.iterative_projection(trial)
                    
                    # Selection with feasibility tracking
                    trial_obj, dv_const, phys_const = self.evaluate_solution(trial, return_components=True)
                    trial_violation = dv_const + phys_const
                    
                    if trial_violation <= self.feasibility_threshold:
                        if trial_obj < best_feasible_score:
                            best_feasible_score = trial_obj
                            best_feasible_solution = trial.copy()
                            improved = True
                    
                    # Selection based on scores
                    trial_score = self.evaluate_population(trial.reshape(1,-1))[0]
                    if trial_score <= scores[i]:
                        population[i] = trial
                        scores[i] = trial_score
                        improved = True
                        
                        # Update best if needed
                        if trial_score < best_score:
                            best_score = trial_score
                            best_solution = trial.copy()
                
                # Check convergence
                if len(history) >= self.update_nlast:
                    if np.std(history[-self.update_nlast:]) < self.tol:
                        logger.info(f"Converged at iteration {iteration}")
                        break
                history.append(best_score)
                
                # Update stall count with improvement tracking
                if not improved or abs(best_score - prev_best) < self.atol:
                    stall_count += 1
                else:
                    stall_count = 0
                    
                if stall_count >= self.stall_generations:
                    logger.info(f"Stopping due to stall at iteration {iteration}")
                    break
                    
                if iteration % 20 == 0:
                    logger.info(f"Iteration {iteration}: Best score = {best_score:.6f}, "
                              f"Best feasible = {best_feasible_score:.6f}")
                    
                iteration += 1
            
            # Return best feasible solution if found, otherwise best overall
            if best_feasible_solution is not None:
                return {
                    'x': best_feasible_solution,
                    'success': True,
                    'message': f"Found feasible solution after {iteration} iterations",
                    'n_iterations': iteration,
                    'n_function_evals': iteration * self.population_size
                }
            else:
                return {
                    'x': best_solution,
                    'success': False,
                    'message': f"No feasible solution found after {iteration} iterations",
                    'n_iterations': iteration,
                    'n_function_evals': iteration * self.population_size
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

    def solve(self, initial_guess, bounds):
        """Solve using enhanced Differential Evolution."""
        try:
            logger.info("Starting DE optimization...")
            start_time = time.time()
            
            result = self.optimize()
            result['execution_time'] = time.time() - start_time
            
            return result
            
        except Exception as e:
            logger.error(f"DE solver failed: {str(e)}")
            return {
                'x': np.zeros(self.n_stages),
                'success': False,
                'message': str(e),
                'execution_time': time.time() - start_time
            }
