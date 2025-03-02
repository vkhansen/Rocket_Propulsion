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
        self.population_size = 30  # Increased for better coverage
        self.mutation_min = 0.3  # Lower bound for mutation
        self.mutation_max = 0.7  # Upper bound for mutation
        self.recombination = 0.9  # Higher recombination for better mixing
        self.strategy = 'best1bin'  # More exploitative strategy
        self.max_iterations = 2000  # Increased iterations for convergence
        self.tol = 1e-6  # Strict tolerance
        self.atol = 1e-6  # Absolute tolerance
        self.update_nlast = 10  # Check last N generations for convergence
        
        # Adaptive parameters
        self.adaptive_penalty = True
        self.penalty_factor = 10.0
        self.penalty_growth = 2.0

    def evaluate_population(self, population):
        """Evaluate population with adaptive penalties."""
        scores = np.zeros(len(population))
        for i, x in enumerate(population):
            try:
                # Get objective and constraint components
                obj, dv_const, phys_const = self.evaluate_solution(x, return_components=True)
                
                # Apply adaptive penalty
                if self.adaptive_penalty:
                    penalty = self.penalty_factor
                    total_violation = dv_const + phys_const
                    if total_violation > 0.1:
                        penalty *= self.penalty_growth
                    scores[i] = obj + penalty * total_violation
                else:
                    scores[i] = obj + self.penalty_factor * (dv_const + phys_const)
                    
            except Exception as e:
                logger.error(f"Error evaluating solution: {str(e)}")
                scores[i] = float('inf')
                
        return scores

    def initialize_population(self):
        """Initialize population using Latin Hypercube Sampling."""
        try:
            from scipy.stats import qmc
            
            # Create Latin Hypercube sampler
            sampler = qmc.LatinHypercube(d=self.n_stages)
            
            # Generate samples in [0,1] space
            samples = sampler.random(n=self.population_size)
            
            # Scale samples to bounds and ensure total delta-v
            population = np.zeros((self.population_size, self.n_stages))
            for i in range(self.population_size):
                # First scale to bounds
                for j in range(self.n_stages):
                    lower, upper = self.bounds[j]
                    population[i,j] = samples[i,j] * (upper - lower) + lower
                
                # Then project to feasible space
                population[i] = super().project_to_feasible(population[i])
                
            return population
            
        except Exception as e:
            logger.warning(f"LHS initialization failed: {str(e)}, using uniform random")
            return self._uniform_random_init()

    def _uniform_random_init(self):
        """Fallback uniform random initialization."""
        population = np.zeros((self.population_size, self.n_stages))
        for i in range(self.population_size):
            for j in range(self.n_stages):
                lower, upper = self.bounds[j]
                population[i,j] = np.random.uniform(lower, upper)
            population[i] = super().project_to_feasible(population[i])
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
            
            # Main optimization loop
            iteration = 0
            stall_count = 0
            history = []
            
            while iteration < self.max_iterations:
                # Store previous best for convergence check
                prev_best = best_score
                
                # Evolve population
                for i in range(self.population_size):
                    # Select parents
                    idxs = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                    
                    # Create trial vector
                    mutation = np.random.uniform(self.mutation_min, self.mutation_max)
                    if self.strategy == 'best1bin':
                        mutant = population[best_idx] + mutation * (b - c)
                    else:
                        mutant = a + mutation * (b - c)
                    
                    # Crossover
                    cross_points = np.random.rand(self.n_stages) < self.recombination
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.n_stages)] = True
                    trial = np.where(cross_points, mutant, population[i])
                    
                    # Project to feasible space
                    trial = super().project_to_feasible(trial)
                    
                    # Selection
                    trial_score = self.evaluate_population(trial.reshape(1,-1))[0]
                    if trial_score <= scores[i]:
                        population[i] = trial
                        scores[i] = trial_score
                        
                        # Update best if needed
                        if trial_score < best_score:
                            best_score = trial_score
                            best_solution = trial.copy()
                
                # Check convergence
                if len(history) >= self.update_nlast:
                    if np.std(history[-self.update_nlast:]) < self.tol:
                        break
                history.append(best_score)
                
                # Update stall count
                if abs(best_score - prev_best) < self.atol:
                    stall_count += 1
                else:
                    stall_count = 0
                    
                if stall_count >= 20:  # Early stopping if stalled
                    break
                    
                iteration += 1
            
            # Return best feasible solution if found, otherwise best overall
            if self.best_feasible is not None:
                return {
                    'x': self.best_feasible,
                    'success': True,
                    'message': f"Iterations: {iteration}, Feasible: {self.n_feasible}, Infeasible: {self.n_infeasible}",
                    'n_iterations': iteration,
                    'n_function_evals': iteration * self.population_size
                }
            else:
                # Check if best solution is feasible
                is_feasible, violation = self.check_feasibility(best_solution)
                return {
                    'x': best_solution,
                    'success': is_feasible,
                    'message': "No feasible solution found" if not is_feasible else "Optimization successful",
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
            start_time = time.time()
            result = self.optimize()
            duration = time.time() - start_time
            
            return self.process_results(
                x=result['x'],
                success=result['success'],
                message=result['message'],
                n_iterations=result['n_iterations'],
                n_function_evals=result['n_function_evals'],
                time=duration
            )
            
        except Exception as e:
            logger.error(f"DE solver failed: {str(e)}")
            return self.process_results(
                x=initial_guess,
                success=False,
                message=str(e),
                n_iterations=0,
                n_function_evals=0,
                time=0.0
            )

    def polish_solution(self, x, violation):
        """Polish promising solutions using L-BFGS-B."""
        if violation > 0.01:  # Only polish low-violation solutions
            return x, self.objective(x)
            
        scales = np.linspace(0.98, 1.02, 5)  # Try multiple scaling factors
        best_x = x
        best_obj = self.objective(x)
        
        for scale in scales:
            try:
                result = minimize(
                    self.objective,
                    x * scale,
                    method='L-BFGS-B',
                    bounds=self.bounds,
                    options={'ftol': 1e-10, 'maxiter': 100}
                )
                if result.success and result.fun < best_obj:
                    best_x = result.x
                    best_obj = result.fun
            except:
                continue
                
        return best_x, best_obj

    def _objective_wrapper(self, x):
        """Wrapper for the objective function to ensure proper mapping."""
        result = self.objective(x)
        if isinstance(result, tuple):
            return float(result[0])  # Return just the objective value
        if result == float('inf') or result > 1e10:  # Handle very large penalties
            return float('inf')
        return float(result)
        
    def get_violation(self, x):
        """Calculate constraint violation."""
        total = np.sum(x)
        violation = abs(total - self.TOTAL_DELTA_V)
        
        # Check bound constraints
        for i, (lower, upper) in enumerate(self.bounds):
            if x[i] < lower:
                violation += abs(x[i] - lower)
            elif x[i] > upper:
                violation += abs(x[i] - upper)
                
        return violation if violation < 1e10 else float('inf')  # Cap very large violations

    def objective(self, x):
        """Objective function with enhanced constraint handling."""
        violation = self.get_violation(x)
        
        if violation > 0.1:  # Major violation
            return 100.0 * violation
        elif violation > 0:  # Minor violation
            return 10.0 * violation
            
        # Calculate payload fraction
        try:
            result = objective_with_penalty(
                x, self.G0, self.ISP, self.EPSILON,
                self.TOTAL_DELTA_V, self.bounds,
                return_tuple=False  # Ensure we get a scalar value
            )
            return result
        except:
            return float('inf')
