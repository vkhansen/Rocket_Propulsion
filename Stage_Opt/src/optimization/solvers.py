"""Optimization solvers."""
import time
import numpy as np
from scipy.optimize import minimize, basinhopping, differential_evolution
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize as pymoo_minimize
from ..utils.config import logger
from .objective import payload_fraction_objective

__all__ = [
    'solve_with_slsqp',
    'solve_with_basin_hopping',
    'solve_with_ga',
    'solve_with_differential_evolution',
    'solve_with_adaptive_ga',
    'solve_with_pso'
]

class RocketOptimizationProblem(Problem):
    """Problem definition for rocket stage optimization."""
    
    def __init__(self, n_var, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V):
        xl = np.array([b[0] for b in bounds])
        xu = np.array([b[1] for b in bounds])
        super().__init__(n_var=n_var, n_obj=1, n_constr=1, xl=xl, xu=xu)
        self.G0 = G0
        self.ISP = ISP
        self.EPSILON = EPSILON
        self.TOTAL_DELTA_V = TOTAL_DELTA_V

    def _evaluate(self, x, out, *args, **kwargs):
        # Handle both single solutions and populations
        x = np.atleast_2d(x)
        
        # Initialize outputs
        n_solutions = x.shape[0]
        f = np.zeros(n_solutions)
        g = np.zeros(n_solutions)
        
        # Evaluate each solution
        for i in range(n_solutions):
            # Calculate payload fraction using correct mass ratio formula
            stage_ratios = []
            for dv, isp, eps in zip(x[i], self.ISP, self.EPSILON):
                # Correct mass ratio formula: λ = exp(-ΔV/(g₀·ISP)) - ε
                ratio = np.exp(-dv / (self.G0 * isp)) - eps
                stage_ratios.append(ratio)
            
            # Store ratios for later use
            if not hasattr(self, '_last_ratios'):
                self._last_ratios = {}
            self._last_ratios[tuple(x[i])] = stage_ratios
            
            f[i] = -np.prod(stage_ratios)  # Negative because we minimize
            
            # Constraint: total delta-v
            g[i] = abs(np.sum(x[i]) - self.TOTAL_DELTA_V)
        
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
        # Get Basin-Hopping parameters from config
        bh_config = config.get('basin_hopping', {})
        n_iterations = bh_config.get('n_iterations', 100)
        temperature = bh_config.get('temperature', 1.0)
        stepsize = bh_config.get('stepsize', 0.5)
        
        # Define the objective function with penalty for constraint violation
        def objective(x):
            # Calculate payload fraction
            stage_ratios = [np.exp(-dv / (G0 * isp)) - eps 
                          for dv, isp, eps in zip(x, ISP, EPSILON)]
            obj_value = -np.prod(stage_ratios)  # Negative because we minimize
            
            # Add penalty for constraint violation
            penalty_weight = 1e6
            delta_v_violation = abs(np.sum(x) - TOTAL_DELTA_V)
            penalty = penalty_weight * delta_v_violation
            
            return obj_value + penalty
        
        # Define bounds as a list of tuples
        bounds_list = [(b[0], b[1]) for b in bounds]
        
        # Run optimization
        result = basinhopping(
            objective,
            initial_guess,
            niter=n_iterations,
            T=temperature,
            stepsize=stepsize,
            minimizer_kwargs={
                'method': 'L-BFGS-B',
                'bounds': bounds_list
            }
        )
        
        if not result.lowest_optimization_result.success:
            logger.warning("Basin-Hopping optimization did not converge")
            return initial_guess
            
        # Scale solution to meet total delta-v constraint exactly
        solution = result.x
        scale = TOTAL_DELTA_V / np.sum(solution)
        return solution * scale
        
    except Exception as e:
        logger.error(f"Basin-Hopping optimization failed: {str(e)}")
        return initial_guess

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
        
        # Create initial population with initial guess
        n_var = len(initial_guess)
        initial_population = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(population_size, n_var)
        )
        initial_population[0] = initial_guess  # Include initial guess
        
        # Create problem
        problem = RocketOptimizationProblem(
            n_var=n_var,
            bounds=bounds,
            G0=G0,
            ISP=ISP,
            EPSILON=EPSILON,
            TOTAL_DELTA_V=TOTAL_DELTA_V
        )
        
        # Setup algorithm
        algorithm = GA(
            pop_size=population_size,
            sampling=initial_population,
            crossover=SBX(prob=crossover_prob, eta=crossover_eta),
            mutation=PM(prob=mutation_prob, eta=mutation_eta)
        )
        
        # Run optimization
        res = pymoo_minimize(
            problem,
            algorithm,
            ('n_gen', n_generations),
            seed=1
        )
        
        if res.X is None:
            logger.warning("GA optimization did not find a solution")
            return initial_guess
            
        # Scale solution to meet total delta-v constraint exactly
        solution = res.X
        scale = TOTAL_DELTA_V / np.sum(solution)
        return solution * scale
        
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
            
            # Ensure fitness values are positive
            if np.any(fitness <= 0):
                logger.warning("Non-positive fitness values encountered. Adjusting to small positive values.")
                fitness = np.clip(fitness, 1e-6, None)
            
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
