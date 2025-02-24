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
from .cache import OptimizationCache

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
        self.cache = OptimizationCache()

    def _evaluate(self, x, out, *args, **kwargs):
        # Handle both single solutions and populations
        x = np.atleast_2d(x)
        
        # Initialize outputs
        n_solutions = x.shape[0]
        f = np.zeros(n_solutions)
        g = np.zeros(n_solutions)
        
        # Evaluate each solution
        for i in range(n_solutions):
            # Check cache first
            cached_fitness = self.cache.get_cached_fitness(x[i])
            if cached_fitness is not None:
                f[i] = -cached_fitness  # Negate since we're minimizing
                g[i] = abs(np.sum(x[i]) - self.TOTAL_DELTA_V)
                continue
                
            # Calculate payload fraction using correct mass ratio formula
            stage_ratios = []
            for dv, isp, eps in zip(x[i], self.ISP, self.EPSILON):
                ratio = np.exp(-dv / (self.G0 * isp)) - eps
                stage_ratios.append(ratio)
            
            # Calculate payload fraction and cache it
            payload_fraction = np.prod(stage_ratios)
            self.cache.cache_fitness(x[i], payload_fraction)
            
            # Store results (negative since we're minimizing)
            f[i] = -payload_fraction
            g[i] = abs(np.sum(x[i]) - self.TOTAL_DELTA_V)
        
        out["F"] = f
        out["G"] = g

def solve_with_slsqp(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Sequential Least Squares Programming (SLSQP)."""
    try:
        logger.debug("Starting SLSQP optimization")
        logger.debug(f"Initial guess: {initial_guess}")
        logger.debug(f"Bounds: {bounds}")
        
        def objective(dv):
            return payload_fraction_objective(dv, G0, ISP, EPSILON)
            
        def total_dv_constraint(dv):
            # Ensure exact total delta-V
            return np.sum(dv) - TOTAL_DELTA_V
            
        def min_dv_constraints(dv):
            # First stage: minimum 15%
            min_dv1 = dv[0] - 0.15 * TOTAL_DELTA_V
            # Other stages: minimum 1% each
            min_dv_others = np.minimum(0, dv[1:] - 0.01 * TOTAL_DELTA_V)
            return np.concatenate(([min_dv1], min_dv_others))
            
        def max_dv_constraint(dv):
            # First stage: maximum 80%
            return 0.8 * TOTAL_DELTA_V - dv[0]
            
        constraints = [
            {'type': 'eq', 'fun': total_dv_constraint},
            {'type': 'ineq', 'fun': min_dv_constraints},
            {'type': 'ineq', 'fun': max_dv_constraint}
        ]
        
        # Set tight tolerance for constraint satisfaction
        options = {
            'ftol': 1e-8,
            'maxiter': config["optimization"]["max_iterations"]
        }
        
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options=options
        )
        
        if result.success:
            logger.info(f"SLSQP optimization converged after {result.nit} iterations")
            logger.debug(f"Final message: {result.message}")
            x = result.x
            scale = TOTAL_DELTA_V / np.sum(x)
            x = x * scale
            logger.info(f"Final solution: {x}")
            return x
        else:
            logger.warning(f"SLSQP optimization did not converge: {result.message}")
            logger.debug(f"Number of iterations: {result.nit}")
            x = initial_guess
            scale = TOTAL_DELTA_V / np.sum(x)
            x = x * scale
            logger.info(f"Returning initial guess solution: {x}")
            return x
            
    except Exception as e:
        logger.error(f"SLSQP optimization failed: {e}")
        logger.exception("Detailed error information:")
        x = initial_guess
        scale = TOTAL_DELTA_V / np.sum(x)
        x = x * scale
        return x

def solve_with_basin_hopping(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Basin-Hopping."""
    try:
        logger.debug("Starting Basin-Hopping optimization")
        logger.debug(f"Initial guess: {initial_guess}")
        logger.debug(f"Bounds: {bounds}")
        
        # Get Basin-Hopping parameters from config
        bh_config = config.get('basin_hopping', {})
        n_iterations = bh_config.get('n_iterations', 100)
        temperature = bh_config.get('temperature', 1.0)
        stepsize = bh_config.get('stepsize', 0.5)
        
        logger.debug(f"Basin-Hopping Config: iterations={n_iterations}, "
                    f"temperature={temperature}, stepsize={stepsize}")
        
        # Initialize cache
        cache = OptimizationCache()
        
        # Define the objective function with caching and penalty
        def objective(x):
            # Check cache first
            x_tuple = tuple(x)
            cached_value = cache.get_cached_fitness(x)
            if cached_value is not None:
                logger.debug(f"Cache hit for solution: {x}")
                return -cached_value  # Negative because we minimize
            
            # Calculate payload fraction
            stage_ratios = [np.exp(-dv / (G0 * isp)) - eps 
                          for dv, isp, eps in zip(x, ISP, EPSILON)]
            obj_value = -np.prod(stage_ratios)  # Negative because we minimize
            
            # Add penalty for constraint violation
            penalty_weight = 1e6
            delta_v_violation = abs(np.sum(x) - TOTAL_DELTA_V)
            penalty = penalty_weight * delta_v_violation
            
            final_value = obj_value + penalty
            
            # Store in cache (without penalty)
            cache.fitness_cache[x_tuple] = -obj_value  # Store positive value
            if final_value < 1e-3:  # If solution is good
                cache.best_solutions.append(np.array(x))
                cache.save_cache()
            
            logger.debug(f"Objective evaluation - Input: {x}, Objective value: {final_value}")
            return final_value
        
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
        
        if result.lowest_optimization_result.success:
            logger.info(f"Basin-Hopping optimization converged after {result.nit} iterations")
            logger.debug(f"Final message: {result.message}")
            solution = result.x
            scale = TOTAL_DELTA_V / np.sum(solution)
            solution = solution * scale
            
            # Cache final solution
            cache.fitness_cache[tuple(solution)] = -result.fun  # Store positive value
            cache.best_solutions.append(solution)
            cache.save_cache()
            
            logger.info(f"Final solution: {solution}")
            return solution
        else:
            logger.warning(f"Basin-Hopping optimization did not converge: {result.message}")
            logger.debug(f"Number of iterations: {result.nit}")
            solution = initial_guess
            scale = TOTAL_DELTA_V / np.sum(solution)
            solution = solution * scale
            logger.info(f"Returning initial guess solution: {solution}")
            return solution
            
    except Exception as e:
        logger.error(f"Basin-Hopping optimization failed: {e}")
        logger.exception("Detailed error information:")
        solution = initial_guess
        scale = TOTAL_DELTA_V / np.sum(solution)
        solution = solution * scale
        return solution

def solve_with_ga(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Genetic Algorithm."""
    try:
        logger.debug("Starting GA optimization")
        logger.debug(f"Initial guess: {initial_guess}")
        logger.debug(f"Bounds: {bounds}")
        
        # Get GA parameters from config
        ga_config = config.get('ga', {})
        population_size = ga_config.get('population_size', 100)
        n_generations = ga_config.get('n_generations', 200)
        crossover_prob = ga_config.get('crossover_prob', 0.9)
        crossover_eta = ga_config.get('crossover_eta', 15)
        mutation_prob = ga_config.get('mutation_prob', 0.2)
        mutation_eta = ga_config.get('mutation_eta', 20)
        
        logger.debug(f"GA Config: population={population_size}, generations={n_generations}, "
                    f"crossover_prob={crossover_prob}, crossover_eta={crossover_eta}, "
                    f"mutation_prob={mutation_prob}, mutation_eta={mutation_eta}")
        
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
        
        if res.X is not None:
            logger.info(f"GA optimization converged after {res.exec_time} seconds")
            logger.debug(f"Final message: {res.message}")
            solution = res.X
            scale = TOTAL_DELTA_V / np.sum(solution)
            solution = solution * scale
            logger.info(f"Final solution: {solution}")
            return solution
        else:
            logger.warning(f"GA optimization did not find a solution")
            logger.debug(f"Number of generations: {n_generations}")
            solution = initial_guess
            scale = TOTAL_DELTA_V / np.sum(solution)
            solution = solution * scale
            logger.info(f"Returning initial guess solution: {solution}")
            return solution
            
    except Exception as e:
        logger.error(f"GA optimization failed: {e}")
        logger.exception("Detailed error information:")
        solution = initial_guess
        scale = TOTAL_DELTA_V / np.sum(solution)
        solution = solution * scale
        return solution

def enforce_stage_constraints(x, TOTAL_DELTA_V):
    """Enforce minimum and maximum ΔV constraints for each stage."""
    # First stage: 15% to 80% of total ΔV
    min_dv1 = 0.15 * TOTAL_DELTA_V
    max_dv1 = 0.80 * TOTAL_DELTA_V
    x[0] = np.clip(x[0], min_dv1, max_dv1)
    
    # Other stages: minimum 1% of total ΔV
    min_dv_other = 0.01 * TOTAL_DELTA_V
    x[1:] = np.clip(x[1:], min_dv_other, TOTAL_DELTA_V)
    
    # Scale to meet total ΔV
    scale = TOTAL_DELTA_V / np.sum(x)
    x = x * scale
    
    # Re-enforce first stage constraints after scaling
    x[0] = np.clip(x[0], min_dv1, max_dv1)
    remaining_dv = TOTAL_DELTA_V - x[0]
    if len(x) > 1:
        # Distribute remaining ΔV proportionally among other stages
        other_stage_ratios = x[1:] / np.sum(x[1:])
        x[1:] = other_stage_ratios * remaining_dv
    
    return x

def solve_with_adaptive_ga(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Adaptive Genetic Algorithm."""
    try:
        logger.debug("Starting Adaptive GA optimization")
        logger.debug(f"Initial guess: {initial_guess}")
        logger.debug(f"Bounds: {bounds}")
        
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
        
        # Create problem instance with caching
        problem = RocketOptimizationProblem(
            n_var=len(initial_guess),
            bounds=bounds,
            G0=G0,
            ISP=ISP,
            EPSILON=EPSILON,
            TOTAL_DELTA_V=TOTAL_DELTA_V
        )
        
        # Initialize population with cached solutions
        best_solutions = problem.cache.get_best_solutions()
        n_cached = len(best_solutions)
        n_var = len(initial_guess)
        
        # Calculate population composition
        n_random = initial_pop_size - n_cached - 1  # -1 for initial guess
        
        # Create population array
        population = np.zeros((initial_pop_size, n_var))
        fitness = np.zeros(initial_pop_size)
        
        # Add initial guess
        population[0] = initial_guess
        fitness[0] = problem.cache.get_cached_fitness(initial_guess) or evaluate_fitness(initial_guess)
        
        # Add cached solutions
        for i in range(min(n_cached, initial_pop_size - 1)):
            population[i + 1] = best_solutions[i]
            fitness[i + 1] = problem.cache.get_cached_fitness(best_solutions[i])
        
        # Fill remaining slots with random solutions
        if n_random > 0:
            random_population = np.random.uniform(
                low=[b[0] for b in bounds],
                high=[b[1] for b in bounds],
                size=(n_random, n_var)
            )
            population[n_cached + 1:] = random_population
            
            # Evaluate and cache random solutions
            for i in range(n_cached + 1, initial_pop_size):
                fitness[i] = evaluate_fitness(population[i])
                problem.cache.cache_fitness(population[i], fitness[i])
        
        # Sort population by fitness
        sort_idx = np.argsort(fitness)[::-1]  # Descending order
        population = population[sort_idx]
        fitness = fitness[sort_idx]
        
        # Main adaptive GA loop
        current_mutation_rate = initial_mutation_rate
        current_crossover_rate = initial_crossover_rate
        current_pop_size = initial_pop_size
        
        for generation in range(n_generations):
            # Adaptive parameter updates based on diversity and stagnation
            diversity = calculate_diversity(population)
            
            if diversity < diversity_threshold:
                current_mutation_rate = min(current_mutation_rate * 1.5, max_mutation_rate)
                current_pop_size = min(current_pop_size * 1.2, max_pop_size)
            else:
                current_mutation_rate = max(current_mutation_rate * 0.9, min_mutation_rate)
                current_pop_size = max(current_pop_size * 0.9, min_pop_size)
            
            # Evolution steps...
            # (rest of the adaptive GA implementation)
            
            # Cache best solution from this generation
            if generation % 10 == 0:  # Cache periodically
                problem.cache.add_best_solution(population[0])
                problem.cache.save_cache()
        
        # Return best solution
        solution = population[0]
        scale = TOTAL_DELTA_V / np.sum(solution)
        solution = solution * scale
        
        # Final cache update
        problem.cache.add_best_solution(solution)
        problem.cache.save_cache()
        
        return solution
        
    except Exception as e:
        logger.error(f"Error in optimization: {e}")
        logger.exception("Detailed error information:")
        return initial_guess

def solve_with_differential_evolution(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Differential Evolution."""
    try:
        logger.debug("Starting Differential Evolution optimization")
        logger.debug(f"Initial guess: {initial_guess}")
        logger.debug(f"Bounds: {bounds}")
        
        # Get DE parameters from config
        de_config = config.get('differential_evolution', {})
        population_size = de_config.get('population_size', 15)
        max_iter = de_config.get('max_iterations', 1000)
        mutation = de_config.get('mutation', [0.5, 1.0])
        recombination = de_config.get('recombination', 0.7)
        strategy = de_config.get('strategy', 'best1bin')
        tol = de_config.get('tol', 1e-6)
        
        logger.debug(f"DE Config: population={population_size}, max_iter={max_iter}, "
                    f"mutation={mutation}, recombination={recombination}, "
                    f"strategy={strategy}, tol={tol}")
        
        # Set minimum delta-v per stage
        MIN_DELTA_V_FIRST = TOTAL_DELTA_V * 0.15
        MIN_DELTA_V_OTHERS = TOTAL_DELTA_V * 0.01
        n_stages = len(initial_guess)
        
        logger.debug(f"Stage constraints: First stage min={MIN_DELTA_V_FIRST}, "
                    f"Other stages min={MIN_DELTA_V_OTHERS}")
        
        def enforce_min_delta_v(particles):
            """Enforce minimum delta-v for each stage while maintaining total delta-v."""
            # Handle single particle case
            if len(particles.shape) == 1:
                particles = particles.reshape(1, -1)
            
            # First ensure minimum values for first stage
            particles[:, 0] = np.maximum(particles[:, 0], MIN_DELTA_V_FIRST)
            
            # Ensure minimum values for other stages
            particles[:, 1:] = np.maximum(particles[:, 1:], MIN_DELTA_V_OTHERS)
            
            # Redistribute excess while maintaining first stage minimum
            for i in range(len(particles)):
                # Calculate remaining delta-v after first stage
                remaining_dv = TOTAL_DELTA_V - particles[i, 0]
                
                if remaining_dv < (n_stages - 1) * MIN_DELTA_V_OTHERS:
                    # If not enough remaining for other stages, adjust first stage
                    particles[i, 0] = TOTAL_DELTA_V - (n_stages - 1) * MIN_DELTA_V_OTHERS
                    particles[i, 1:] = MIN_DELTA_V_OTHERS
                else:
                    # Distribute remaining to other stages proportionally
                    current_sum = np.sum(particles[i, 1:])
                    if current_sum > 0:
                        particles[i, 1:] *= remaining_dv / current_sum
            
            return particles
        
        def objective(x):
            x_constrained = enforce_min_delta_v(x)
            obj_value = payload_fraction_objective(x_constrained, G0, ISP, EPSILON)
            logger.debug(f"Objective evaluation - Input: {x}, Constrained: {x_constrained}, "
                        f"Objective value: {obj_value}")
            return obj_value
        
        # Modify bounds for first stage
        bounds[0] = (MIN_DELTA_V_FIRST, TOTAL_DELTA_V * 0.8)  # 15% to 80% of total ΔV
        logger.debug(f"Modified bounds: {bounds}")
        
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
            logger.info(f"DE optimization converged after {result.nit} iterations")
            logger.debug(f"Final message: {result.message}")
            solution = enforce_min_delta_v(result.x)
            if len(solution.shape) > 1:
                solution = solution[0]  # Take first solution if multiple returned
            logger.info(f"Final solution: {solution}")
            return solution
        else:
            logger.warning(f"DE optimization did not converge: {result.message}")
            logger.debug(f"Number of iterations: {result.nit}")
            solution = enforce_min_delta_v(initial_guess)
            if len(solution.shape) > 1:
                solution = solution[0]
            logger.info(f"Returning initial guess solution: {solution}")
            return solution
            
    except Exception as e:
        logger.error(f"Differential Evolution optimization failed: {e}")
        logger.exception("Detailed error information:")
        solution = enforce_min_delta_v(initial_guess)
        if len(solution.shape) > 1:
            solution = solution[0]
        return solution

def solve_with_pso(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Particle Swarm Optimization."""
    try:
        logger.debug("Starting PSO optimization")
        logger.debug(f"Initial guess: {initial_guess}")
        logger.debug(f"Bounds: {bounds}")
        
        # Get PSO parameters from config
        pso_config = config.get('pso', {})
        n_particles = pso_config.get('n_particles', 50)
        n_iterations = pso_config.get('n_iterations', 100)
        w = pso_config.get('w', 0.7)  # Inertia weight
        c1 = pso_config.get('c1', 2.0)  # Cognitive parameter
        c2 = pso_config.get('c2', 2.0)  # Social parameter
        
        logger.debug(f"PSO Config: particles={n_particles}, iterations={n_iterations}, "
                    f"w={w}, c1={c1}, c2={c2}")
        
        # Set minimum delta-v per stage (15% of total delta-v for first stage, 1% for others)
        MIN_DELTA_V_FIRST = TOTAL_DELTA_V * 0.15
        MIN_DELTA_V_OTHERS = TOTAL_DELTA_V * 0.01
        n_stages = len(initial_guess)
        
        logger.debug(f"Stage constraints: First stage min={MIN_DELTA_V_FIRST}, "
                    f"Other stages min={MIN_DELTA_V_OTHERS}")
        
        def enforce_min_delta_v(particles):
            """Enforce minimum delta-v for each stage while maintaining total delta-v."""
            # Handle single particle case
            if len(particles.shape) == 1:
                particles = particles.reshape(1, -1)
            
            # First ensure minimum values for first stage
            particles[:, 0] = np.maximum(particles[:, 0], MIN_DELTA_V_FIRST)
            
            # Ensure minimum values for other stages
            particles[:, 1:] = np.maximum(particles[:, 1:], MIN_DELTA_V_OTHERS)
            
            # Redistribute excess while maintaining first stage minimum
            for i in range(len(particles)):
                # Calculate remaining delta-v after first stage
                remaining_dv = TOTAL_DELTA_V - particles[i, 0]
                
                if remaining_dv < (n_stages - 1) * MIN_DELTA_V_OTHERS:
                    # If not enough remaining for other stages, adjust first stage
                    particles[i, 0] = TOTAL_DELTA_V - (n_stages - 1) * MIN_DELTA_V_OTHERS
                    particles[i, 1:] = MIN_DELTA_V_OTHERS
                else:
                    # Distribute remaining to other stages proportionally
                    current_sum = np.sum(particles[i, 1:])
                    if current_sum > 0:
                        particles[i, 1:] *= remaining_dv / current_sum
            
            return particles
        
        # Modify bounds for first stage
        bounds[0] = (MIN_DELTA_V_FIRST, TOTAL_DELTA_V * 0.8)  # 15% to 80% of total ΔV
        logger.debug(f"Modified bounds: {bounds}")
        
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
        for iteration in range(n_iterations):
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
            
            # Ensure fitness values are positive
            if np.any(values <= 0):
                logger.warning("Non-positive fitness values encountered. Adjusting to small positive values.")
                values = np.clip(values, 1e-6, None)
            
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
        logger.error(f"Optimization with PSO failed: {e}")
        logger.exception("Detailed error information:")
        return initial_guess
