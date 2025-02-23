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
        # Get GA parameters from config
        ga_config = config.get('ga', {})
        population_size = ga_config.get('population_size', 100)
        n_generations = ga_config.get('n_generations', 200)
        crossover_prob = ga_config.get('crossover_prob', 0.9)
        crossover_eta = ga_config.get('crossover_eta', 15)
        mutation_prob = ga_config.get('mutation_prob', 0.2)
        mutation_eta = ga_config.get('mutation_eta', 20)
        
        # Create problem
        problem = RocketOptimizationProblem(
            n_var=len(initial_guess),
            bounds=bounds,
            G0=G0,
            ISP=ISP,
            EPSILON=EPSILON,
            TOTAL_DELTA_V=TOTAL_DELTA_V
        )
        
        # Setup algorithm
        algorithm = GA(
            pop_size=population_size,
            sampling=initial_guess,
            crossover_prob=crossover_prob,
            crossover_eta=crossover_eta,
            mutation_prob=mutation_prob,
            mutation_eta=mutation_eta
        )
        
        # Run optimization
        result = minimize(
            problem,
            algorithm,
            ('n_gen', n_generations),
            seed=1,
            verbose=False
        )
        
        return result.X
        
    except Exception as e:
        logger.error(f"GA optimization failed: {str(e)}")
        return initial_guess

def solve_with_adaptive_ga(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Adaptive Genetic Algorithm."""
    try:
        # Get Adaptive GA parameters from config
        ada_ga_config = config.get('adaptive_ga', {})
        initial_pop_size = ada_ga_config.get('initial_pop_size', 100)
        max_pop_size = ada_ga_config.get('max_pop_size', 200)
        min_pop_size = ada_ga_config.get('min_pop_size', 50)
        initial_mutation_rate = ada_ga_config.get('initial_mutation_rate', 0.1)
        max_mutation_rate = ada_ga_config.get('max_mutation_rate', 0.3)
        min_mutation_rate = ada_ga_config.get('min_mutation_rate', 0.01)
        initial_crossover_rate = ada_ga_config.get('initial_crossover_rate', 0.8)
        max_crossover_rate = ada_ga_config.get('max_crossover_rate', 0.95)
        min_crossover_rate = ada_ga_config.get('min_crossover_rate', 0.6)
        diversity_threshold = ada_ga_config.get('diversity_threshold', 0.1)
        stagnation_threshold = ada_ga_config.get('stagnation_threshold', 10)
        n_generations = ada_ga_config.get('n_generations', 200)
        elite_size = ada_ga_config.get('elite_size', 2)
        
        population = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(initial_pop_size, len(initial_guess))
        )
        
        # Add initial guess to population
        population[0] = initial_guess
        
        best_solution = initial_guess
        best_fitness = payload_fraction_objective(initial_guess, G0, ISP, EPSILON)
        stagnation_counter = 0
        current_pop_size = initial_pop_size
        current_mutation_rate = initial_mutation_rate
        current_crossover_rate = initial_crossover_rate
        
        for generation in range(n_generations):
            # Evaluate fitness
            fitness = np.array([payload_fraction_objective(ind, G0, ISP, EPSILON) for ind in population])
            
            # Update best solution
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fitness:
                best_fitness = fitness[min_idx]
                best_solution = population[min_idx].copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Adaptive population size
            if stagnation_counter >= stagnation_threshold:
                current_pop_size = min(max_pop_size, current_pop_size + 10)
                current_mutation_rate = min(max_mutation_rate, current_mutation_rate * 1.2)
                stagnation_counter = 0
            else:
                current_pop_size = max(min_pop_size, current_pop_size - 5)
                current_mutation_rate = max(min_mutation_rate, current_mutation_rate * 0.9)
            
            # Selection
            parents_idx = np.random.choice(
                len(population),
                size=current_pop_size,
                p=1/fitness/np.sum(1/fitness)
            )
            parents = population[parents_idx]
            
            # Crossover
            offspring = parents.copy()
            for i in range(0, len(offspring)-1, 2):
                if np.random.random() < current_crossover_rate:
                    alpha = np.random.random()
                    offspring[i] = alpha * parents[i] + (1-alpha) * parents[i+1]
                    offspring[i+1] = (1-alpha) * parents[i] + alpha * parents[i+1]
            
            # Mutation
            mutation_mask = np.random.random(offspring.shape) < current_mutation_rate
            mutation = np.random.normal(0, 0.1, offspring.shape)
            offspring[mutation_mask] += mutation[mutation_mask]
            
            # Enforce bounds and constraints
            for j in range(len(initial_guess)):
                offspring[:, j] = np.clip(offspring[:, j], bounds[j][0], bounds[j][1])
            
            # Scale to meet total delta-v constraint
            sums = np.sum(offspring, axis=1)
            scale_factors = TOTAL_DELTA_V / sums
            offspring = offspring * scale_factors.reshape(-1, 1)
            
            # Elitism
            if elite_size > 0:
                elite_idx = np.argsort(fitness)[:elite_size]
                offspring[:elite_size] = population[elite_idx]
            
            population = offspring[:current_pop_size]
            
            # Check convergence
            if np.std(fitness) < 1e-6:
                break
        
        return best_solution
        
    except Exception as e:
        logger.error(f"Error in optimization: {str(e)}")
        return initial_guess

def solve_with_differential_evolution(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Differential Evolution."""
    try:
        # Get DE parameters from config
        de_config = config.get('differential_evolution', {})
        population_size = de_config.get('population_size', 15)
        max_iter = de_config.get('max_iterations', 1000)
        mutation = de_config.get('mutation', [0.5, 1.0])
        recombination = de_config.get('recombination', 0.7)
        strategy = de_config.get('strategy', 'best1bin')
        tol = de_config.get('tol', 1e-6)
        
        def objective(x):
            # Scale to meet total delta-v constraint
            scale = TOTAL_DELTA_V / np.sum(x)
            x_scaled = x * scale
            return payload_fraction_objective(x_scaled, G0, ISP, EPSILON)
        
        result = differential_evolution(
            objective,
            bounds=bounds,
            strategy=strategy,
            maxiter=max_iter,
            popsize=population_size,
            mutation=mutation,
            recombination=recombination,
            tol=tol,
            init='sobol'
        )
        
        if result.success:
            # Scale solution to meet total delta-v constraint
            scale = TOTAL_DELTA_V / np.sum(result.x)
            return result.x * scale
        else:
            logger.warning(f"DE optimization did not converge: {result.message}")
            return initial_guess
            
    except Exception as e:
        logger.error(f"Differential Evolution optimization failed: {str(e)}")
        return initial_guess

def solve_with_pso(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Particle Swarm Optimization."""
    try:
        n_particles = config.get('n_particles', 50)
        max_iter = config.get('max_iter', 100)
        w = config.get('w', 0.7)  # Inertia weight
        c1 = config.get('c1', 2.0)  # Cognitive parameter
        c2 = config.get('c2', 2.0)  # Social parameter
        
        # Set minimum delta-v per stage (1% of total delta-v)
        MIN_DELTA_V = TOTAL_DELTA_V * 0.01
        n_stages = len(initial_guess)
        
        def enforce_min_delta_v(particles):
            """Enforce minimum delta-v for each stage while maintaining total delta-v."""
            # First ensure minimum values
            particles = np.maximum(particles, MIN_DELTA_V)
            
            # Then scale to meet total delta-v constraint
            sums = np.sum(particles, axis=1)
            scale_factors = TOTAL_DELTA_V / sums
            particles = particles * scale_factors.reshape(-1, 1)
            
            # If scaling made any values too small, redistribute the excess
            below_min_mask = particles < MIN_DELTA_V
            while np.any(below_min_mask):
                # Set values below minimum to minimum
                particles[below_min_mask] = MIN_DELTA_V
                
                # Recalculate totals and scale remaining values
                for i in range(len(particles)):
                    below_min = below_min_mask[i]
                    if np.any(below_min):
                        # Calculate remaining delta-v to distribute
                        remaining_dv = TOTAL_DELTA_V - np.sum(particles[i][below_min])
                        # Get indices not at minimum
                        above_min = ~below_min
                        if np.any(above_min):
                            # Distribute remaining dv proportionally
                            current_sum = np.sum(particles[i][above_min])
                            if current_sum > 0:
                                scale = remaining_dv / current_sum
                                particles[i][above_min] *= scale
                
                # Update mask for next iteration
                below_min_mask = particles < MIN_DELTA_V
                
                # Break if we can't satisfy constraints
                if np.all(below_min_mask):
                    particles = np.full_like(particles, TOTAL_DELTA_V / n_stages)
                    break
            
            return particles
        
        # Initialize particles
        particles = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(n_particles, len(initial_guess))
        )
        
        # Add initial guess as one of the particles
        particles[0] = initial_guess
        
        # Initialize velocities and best positions
        velocities = np.zeros_like(particles)
        particles = enforce_min_delta_v(particles)
        personal_best_pos = particles.copy()
        personal_best_val = np.array([payload_fraction_objective(p, G0, ISP, EPSILON) for p in particles])
        global_best_idx = np.argmin(personal_best_val)
        global_best_pos = personal_best_pos[global_best_idx].copy()
        global_best_val = personal_best_val[global_best_idx]
        
        # Main PSO loop
        for iteration in range(max_iter):
            # Update velocities
            r1, r2 = np.random.rand(2, n_particles, len(initial_guess))
            velocities = (w * velocities +
                        c1 * r1 * (personal_best_pos - particles) +
                        c2 * r2 * (global_best_pos - particles))
            
            # Update positions
            particles += velocities
            
            # Clip to bounds and enforce constraints
            for j in range(len(initial_guess)):
                particles[:, j] = np.clip(particles[:, j], bounds[j][0], bounds[j][1])
            
            particles = enforce_min_delta_v(particles)
            
            # Update personal and global bests
            values = np.array([payload_fraction_objective(p, G0, ISP, EPSILON) for p in particles])
            improved = values < personal_best_val
            personal_best_pos[improved] = particles[improved]
            personal_best_val[improved] = values[improved]
            
            # Update global best
            min_idx = np.argmin(values)
            if values[min_idx] < global_best_val:
                global_best_val = values[min_idx]
                global_best_pos = particles[min_idx].copy()
            
            # Check convergence
            if iteration > 10 and np.std(values) < 1e-6:
                logger.info(f"PSO converged after {iteration} iterations")
                break
        
        return global_best_pos
        
    except Exception as e:
        logger.error(f"Optimization with PSO failed: {str(e)}")
        return initial_guess
