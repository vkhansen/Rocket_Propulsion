"""Optimization solvers."""
import time
import numpy as np
from scipy.optimize import minimize, basinhopping, differential_evolution
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.core.repair import Repair
from pymoo.termination.default import DefaultSingleObjectiveTermination

from .objective import payload_fraction_objective, objective_with_penalty
from ..utils.config import logger
from ..utils.data import calculate_mass_ratios, calculate_payload_fraction

__all__ = [
    'solve_with_slsqp',
    'solve_with_basin_hopping',
    'solve_with_differential_evolution',
    'solve_with_ga',
    'solve_with_adaptive_ga',
    'solve_with_pso'
]

class DeltaVRepair(Repair):
    """Repair operator to ensure delta-v sum constraint."""
    def __init__(self, total_delta_v):
        super().__init__()
        self.total_delta_v = total_delta_v

    def _do(self, problem, X, **kwargs):
        """Repair the solution to meet the total delta-v constraint."""
        X = np.maximum(X, 0)  # Ensure non-negative values
        sums = np.sum(X, axis=1)
        scale = self.total_delta_v / sums
        X = X * scale[:, None]
        return X

class RocketOptimizationProblem(Problem):
    """Problem definition for rocket stage optimization."""
    def __init__(self, n_var, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V):
        super().__init__(
            n_var=n_var,
            n_obj=1,
            n_constr=1,
            xl=np.array([b[0] for b in bounds]),
            xu=np.array([b[1] for b in bounds])
        )
        self.G0 = G0
        self.ISP = ISP
        self.EPSILON = EPSILON
        self.TOTAL_DELTA_V = TOTAL_DELTA_V

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate the objective and constraints."""
        f = np.array([payload_fraction_objective(dv, self.G0, self.ISP, self.EPSILON) for dv in x])
        g = np.array([np.sum(dv) - self.TOTAL_DELTA_V for dv in x])
        out["F"] = f
        out["G"] = g

def solve_with_slsqp(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Sequential Least Squares Programming (SLSQP)."""
    try:
        def objective(dv):
            return payload_fraction_objective(dv, G0, ISP, EPSILON)
            
        def constraint(dv):
            return float(np.sum(dv) - TOTAL_DELTA_V)
            
        constraints = {'type': 'eq', 'fun': constraint}
        
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={
                'ftol': config["optimization"]["tolerance"],
                'maxiter': config["optimization"]["max_iterations"]
            }
        )
        
        if not result.success:
            logger.warning(f"SLSQP optimization warning: {result.message}")
            
        return result.x
        
    except Exception as e:
        logger.error(f"SLSQP optimization failed: {e}")
        raise

def solve_with_basin_hopping(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Basin-Hopping."""
    try:
        def objective(dv):
            return objective_with_penalty(dv, G0, ISP, EPSILON, TOTAL_DELTA_V)
        
        minimizer_kwargs = {
            "method": "L-BFGS-B",
            "bounds": bounds,
            "options": {
                'ftol': config["optimization"]["tolerance"],
                'maxiter': config["optimization"]["max_iterations"]
            }
        }
        
        result = basinhopping(
            objective,
            initial_guess,
            minimizer_kwargs=minimizer_kwargs,
            niter=config["optimization"]["max_iterations"],
            T=1.0,
            stepsize=0.5,
            interval=50,  # Adjust temperature every 50 steps
            niter_success=10  # Stop after 10 successive successes
        )
        
        if not result.lowest_optimization_result.success:
            logger.warning(f"Basin-hopping optimization warning: {result.message}")
            
        return result.x
        
    except Exception as e:
        logger.error(f"Basin-Hopping optimization failed: {e}")
        raise

def solve_with_ga(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Genetic Algorithm."""
    try:
        class RepairDeltaV(Repair):
            def _do(self, problem, X, **kwargs):
                if len(X.shape) == 1:
                    X = X.reshape(1, -1)
                    
                for i in range(len(X)):
                    x = X[i]
                    current_sum = np.sum(x)
                    if abs(current_sum - TOTAL_DELTA_V) > 1e-6:
                        # Scale the solution to match total ΔV
                        x = x * (TOTAL_DELTA_V / current_sum)
                        # Ensure bounds are satisfied
                        x = np.clip(x, problem.xl, problem.xu)
                        # Re-normalize if clipping changed the sum
                        current_sum = np.sum(x)
                        if abs(current_sum - TOTAL_DELTA_V) > 1e-6:
                            x = x * (TOTAL_DELTA_V / current_sum)
                        X[i] = x
                return X

        class OptimizationProblem(Problem):
            def __init__(self, n_var, n_obj, xl, xu):
                super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)
                self.total_delta_v = None
                self.G0 = None
                self.ISP = None
                self.EPSILON = None

            def _evaluate(self, x, out, *args, **kwargs):
                f = []
                for dv in x:
                    # Calculate payload fraction
                    stage_ratios = [np.exp(-dvi / (self.G0 * isp)) - eps 
                                  for dvi, isp, eps in zip(dv, self.ISP, self.EPSILON)]
                    payload = -np.prod(stage_ratios)  # Negative because GA minimizes
                    
                    # Add penalty for total ΔV constraint
                    penalty = 1e6 * abs(np.sum(dv) - self.total_delta_v)
                    f.append(payload + penalty)
                    
                out["F"] = np.column_stack([f])

        # Convert bounds to numpy array
        bounds = np.array(bounds)
        
        problem = OptimizationProblem(
            n_var=len(initial_guess),
            n_obj=1,
            xl=bounds[:, 0],
            xu=bounds[:, 1]
        )

        problem.total_delta_v = TOTAL_DELTA_V
        problem.G0 = G0
        problem.ISP = ISP
        problem.EPSILON = EPSILON

        ga_config = config["optimization"]["ga"]
        algorithm = GA(
            pop_size=ga_config["population_size"],
            eliminate_duplicates=True,
            mutation=PolynomialMutation(prob=ga_config["mutation_prob"], eta=ga_config["mutation_eta"]),
            crossover=SBX(prob=ga_config["crossover_prob"], eta=ga_config["crossover_eta"]),
            repair=RepairDeltaV()
        )

        termination = DefaultSingleObjectiveTermination(
            xtol=1e-6,
            cvtol=1e-6,
            ftol=1e-6,
            period=20,
            n_max_gen=ga_config["n_generations"],
            n_max_evals=None
        )
        
        res = pymoo_minimize(
            problem,
            algorithm,
            termination,
            seed=1,
            verbose=False
        )

        if res.X is None or not res.success:
            logger.warning(f"GA optimization warning: {res.message}")
            return initial_guess
            
        return res.X
        
    except Exception as e:
        logger.error(f"GA optimization failed: {e}")
        return initial_guess

def solve_with_differential_evolution(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Differential Evolution."""
    try:
        def objective(x):
            """Objective function with penalty for DE."""
            x = np.asarray(x).flatten()
            # Calculate payload fraction
            mass_ratios = calculate_mass_ratios(x, ISP, EPSILON, G0)
            payload_fraction = calculate_payload_fraction(mass_ratios)
            
            # Add penalty for total delta-V constraint
            penalty = 1e6 * abs(np.sum(x) - TOTAL_DELTA_V)
            
            # Additional penalty for solutions close to physical limits
            for ratio in mass_ratios:
                if ratio <= 0.1:
                    penalty += 100.0 * (0.1 - ratio) ** 2
            
            return -payload_fraction + penalty  # Negative because DE minimizes
        
        # Run differential evolution
        de_config = config["optimization"]["differential_evolution"]
        result = differential_evolution(
            objective,
            bounds,
            strategy=de_config["strategy"],
            maxiter=de_config["max_iterations"],
            popsize=de_config["population_size"],
            tol=de_config["tol"],
            mutation=de_config["mutation"],
            recombination=de_config["recombination"],
            seed=None,
            disp=False,
            polish=True,
            updating='immediate'
        )
        
        if not result.success:
            logger.warning(f"Differential Evolution warning: {result.message}")
        
        return result.x
        
    except Exception as e:
        logger.error(f"Differential Evolution optimization failed: {e}")
        return initial_guess

def solve_with_adaptive_ga(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Adaptive Genetic Algorithm."""
    try:
        class AdaptiveGA:
            def __init__(self, config, bounds):
                aga_config = config["optimization"]["adaptive_ga"]
                self.pop_size = aga_config["initial_pop_size"]
                self.max_pop_size = aga_config["max_pop_size"]
                self.min_pop_size = aga_config["min_pop_size"]
                self.mutation_rate = aga_config["initial_mutation_rate"]
                self.max_mutation_rate = aga_config["max_mutation_rate"]
                self.min_mutation_rate = aga_config["min_mutation_rate"]
                self.crossover_rate = aga_config["initial_crossover_rate"]
                self.max_crossover_rate = aga_config["max_crossover_rate"]
                self.min_crossover_rate = aga_config["min_crossover_rate"]
                self.diversity_threshold = aga_config["diversity_threshold"]
                self.stagnation_threshold = aga_config["stagnation_threshold"]
                self.n_generations = aga_config["n_generations"]
                self.elite_size = aga_config["elite_size"]
                self.bounds = bounds
                self.history = []
                self.n_vars = len(bounds)
                self.bounds_low = bounds[:, 0]
                self.bounds_high = bounds[:, 1]
                self.best_fitness = float('inf')
                self.stagnation_counter = 0
                
            def initialize_population(self):
                """Initialize population with random solutions."""
                return np.random.uniform(
                    self.bounds_low, 
                    self.bounds_high, 
                    size=(self.pop_size, self.n_vars)
                )
                
            def repair_solution(self, solution):
                """Repair solution to satisfy constraints."""
                total = np.sum(solution)
                if abs(total - TOTAL_DELTA_V) > 1e-6:
                    solution = solution * (TOTAL_DELTA_V / total)
                    solution = np.clip(solution, self.bounds_low, self.bounds_high)
                    # Re-normalize if clipping changed the sum
                    total = np.sum(solution)
                    if abs(total - TOTAL_DELTA_V) > 1e-6:
                        solution = solution * (TOTAL_DELTA_V / total)
                return solution
                
            def evaluate_population(self, population):
                """Evaluate fitness of population."""
                fitness = []
                for solution in population:
                    mass_ratios = calculate_mass_ratios(solution, ISP, EPSILON, G0)
                    payload_fraction = calculate_payload_fraction(mass_ratios)
                    penalty = 1e6 * abs(np.sum(solution) - TOTAL_DELTA_V)
                    fitness.append(-payload_fraction + penalty)  # Negative because we minimize
                return np.array(fitness)
                
            def select_parents(self, population, fitness):
                """Tournament selection."""
                tournament_size = 3
                idx1 = np.random.randint(0, len(population), tournament_size)
                idx2 = np.random.randint(0, len(population), tournament_size)
                parent1 = population[idx1[np.argmin(fitness[idx1])]]
                parent2 = population[idx2[np.argmin(fitness[idx2])]]
                return parent1, parent2
                
            def crossover(self, parent1, parent2):
                """Adaptive arithmetic crossover."""
                if np.random.random() > self.crossover_rate:
                    return parent1.copy(), parent2.copy()
                    
                # Calculate adaptive crossover strength
                crossover_strength = (self.bounds_high - self.bounds_low) * 0.1 * \
                                  (1.0 - len(self.history) / self.n_generations)
                
                # Apply crossover with varying strength per dimension
                alpha = np.random.rand(self.n_vars)
                child1 = alpha * parent1 + (1 - alpha) * parent2
                child2 = (1 - alpha) * parent1 + alpha * parent2
                
                return child1, child2
                
            def mutate(self, solution):
                """Adaptive Gaussian mutation."""
                if np.random.random() > self.mutation_rate:
                    return solution
                    
                # Calculate adaptive mutation strength
                mutation_strength = (self.bounds_high - self.bounds_low) * 0.1 * \
                                 (1.0 - len(self.history) / self.n_generations)
                
                # Apply mutation
                mutation = np.random.normal(0, mutation_strength, self.n_vars)
                mutated = solution + mutation
                mutated = np.clip(mutated, self.bounds_low, self.bounds_high)
                return mutated
                
            def update_parameters(self, population, fitness):
                """Update adaptive parameters."""
                # Calculate population diversity
                mean_solution = np.mean(population, axis=0)
                diversity = np.mean([np.linalg.norm(sol - mean_solution) for sol in population])
                normalized_diversity = diversity / np.linalg.norm(self.bounds_high - self.bounds_low)
                
                # Update mutation rate based on diversity
                if normalized_diversity < self.diversity_threshold:
                    self.mutation_rate = min(self.max_mutation_rate,
                                          self.mutation_rate * 1.1)
                else:
                    self.mutation_rate = max(self.min_mutation_rate,
                                          self.mutation_rate * 0.9)
                
                # Update crossover rate based on fitness improvement
                if len(self.history) > 0 and min(fitness) < self.best_fitness:
                    self.crossover_rate = min(self.max_crossover_rate,
                                           self.crossover_rate * 1.1)
                    self.stagnation_counter = 0
                else:
                    self.crossover_rate = max(self.min_crossover_rate,
                                           self.crossover_rate * 0.9)
                    self.stagnation_counter += 1
                
                # Update population size based on stagnation
                if self.stagnation_counter >= self.stagnation_threshold:
                    self.pop_size = max(self.min_pop_size,
                                     int(self.pop_size * 0.9))
                    self.stagnation_counter = 0
                
                # Record best fitness
                self.best_fitness = min(min(fitness), self.best_fitness)
                
            def optimize(self):
                """Main optimization loop."""
                population = self.initialize_population()
                
                for gen in range(self.n_generations):
                    # Evaluate population
                    fitness = self.evaluate_population(population)
                    best_idx = np.argmin(fitness)
                    self.history.append(fitness[best_idx])
                    
                    # Update adaptive parameters
                    self.update_parameters(population, fitness)
                    
                    # Create new population
                    new_population = []
                    
                    # Elitism
                    elite_idx = np.argsort(fitness)[:self.elite_size]
                    new_population.extend(population[elite_idx])
                    
                    # Generate offspring
                    while len(new_population) < self.pop_size:
                        # Select parents
                        parent1, parent2 = self.select_parents(population, fitness)
                        
                        # Crossover
                        child1, child2 = self.crossover(parent1, parent2)
                        
                        # Mutation
                        child1 = self.mutate(child1)
                        child2 = self.mutate(child2)
                        
                        # Repair
                        child1 = self.repair_solution(child1)
                        child2 = self.repair_solution(child2)
                        
                        new_population.extend([child1, child2])
                    
                    # Trim population to current size
                    population = np.array(new_population[:self.pop_size])
                
                # Return best solution
                final_fitness = self.evaluate_population(population)
                best_idx = np.argmin(final_fitness)
                return population[best_idx], {'history': self.history}

        # Run optimization
        optimizer = AdaptiveGA(config, np.array(bounds))
        optimal_dv, result = optimizer.optimize()
        
        return optimal_dv

    except Exception as e:
        logger.error(f"Error in optimization: {e}")
        return initial_guess

def solve_with_pso(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Particle Swarm Optimization."""
    n_particles = config.get('n_particles', 50)
    max_iter = config.get('max_iter', 100)
    w = config.get('w', 0.7)  # Inertia weight
    c1 = config.get('c1', 2.0)  # Cognitive parameter
    c2 = config.get('c2', 2.0)  # Social parameter
    
    # Set minimum delta-v per stage (1% of total delta-v)
    MIN_DELTA_V = TOTAL_DELTA_V * 0.01
    
    def enforce_min_delta_v(particles):
        """Enforce minimum delta-v for each stage while maintaining total delta-v."""
        n_stages = particles.shape[1]
        
        # First ensure minimum values
        particles = np.maximum(particles, MIN_DELTA_V)
        
        # Then scale to meet total delta-v constraint
        sums = np.sum(particles, axis=1)
        scale_factors = (TOTAL_DELTA_V / sums).reshape(-1, 1)
        particles *= scale_factors
        
        # If scaling made any values too small, redistribute the excess
        while np.any(particles < MIN_DELTA_V):
            below_min = particles < MIN_DELTA_V
            excess_needed = MIN_DELTA_V - particles[below_min]
            particles[below_min] = MIN_DELTA_V
            
            # Calculate how much we need to reduce from other stages
            for i in range(particles.shape[0]):
                row_below_min = below_min[i]
                if np.any(row_below_min):
                    # Get indices not below minimum
                    above_min_idx = ~row_below_min
                    if np.sum(above_min_idx) > 0:
                        # Calculate total excess needed for this particle
                        total_excess = np.sum(excess_needed[i * n_stages:(i + 1) * n_stages][row_below_min])
                        # Reduce proportionally from stages above minimum
                        reduction_per_stage = total_excess / np.sum(above_min_idx)
                        particles[i][above_min_idx] -= reduction_per_stage
        
        return particles
    
    # Initialize particles
    particles = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(n_particles, len(initial_guess))
    )
    
    # Add initial guess as one of the particles
    particles[0] = initial_guess
    
    # Enforce minimum delta-v and scale to meet constraint
    particles = enforce_min_delta_v(particles)
    
    velocities = np.zeros_like(particles)
    personal_best_pos = particles.copy()
    personal_best_val = np.array([payload_fraction_objective(p, G0, ISP, EPSILON) for p in particles])
    global_best_pos = personal_best_pos[np.argmin(personal_best_val)]
    global_best_val = np.min(personal_best_val)
    
    for iteration in range(max_iter):
        # Update velocities and positions
        r1, r2 = np.random.rand(2, n_particles, len(initial_guess))
        velocities = (w * velocities +
                    c1 * r1 * (personal_best_pos - particles) +
                    c2 * r2 * (global_best_pos - particles))
        
        particles += velocities
        
        # Clip to bounds
        for j in range(len(initial_guess)):
            particles[:, j] = np.clip(particles[:, j], bounds[j][0], bounds[j][1])
        
        # Enforce minimum delta-v and scale to meet constraint
        particles = enforce_min_delta_v(particles)
        
        # Update personal and global bests
        values = np.array([payload_fraction_objective(p, G0, ISP, EPSILON) for p in particles])
        improved = values < personal_best_val
        personal_best_pos[improved] = particles[improved]
        personal_best_val[improved] = values[improved]
        
        min_val_idx = np.argmin(values)
        if values[min_val_idx] < global_best_val:
            global_best_val = values[min_val_idx]
            global_best_pos = particles[min_val_idx].copy()
            
        # Check convergence
        if iteration > 10 and np.std(values) < 1e-6:
            logger.info(f"PSO converged after {iteration} iterations")
            break
    
    return global_best_pos
