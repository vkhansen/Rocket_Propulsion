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
from .objective import payload_fraction_objective, calculate_mass_ratios, calculate_payload_fraction, calculate_stage_ratios
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
    
    def __init__(self, n_var, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config=None):
        """Initialize the optimization problem.
        
        Args:
            n_var: Number of variables (stages)
            bounds: List of (min, max) bounds for each variable
            G0: Gravitational constant
            ISP: List of specific impulse values
            EPSILON: List of structural fraction values
            TOTAL_DELTA_V: Total required delta-v
            config: Configuration dictionary
        """
        xl = np.array([b[0] for b in bounds])
        xu = np.array([b[1] for b in bounds])
        super().__init__(n_var=n_var, n_obj=1, n_constr=1, xl=xl, xu=xu)
        
        self.G0 = G0
        self.ISP = np.asarray(ISP, dtype=float)
        self.EPSILON = np.asarray(EPSILON, dtype=float)
        self.TOTAL_DELTA_V = TOTAL_DELTA_V
        self.config = config if config is not None else {}
        self.cache = OptimizationCache()
    
    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate solutions.
        
        Args:
            x: Solution or population of solutions
            out: Output dictionary for fitness and constraints
        """
        # Evaluate each solution
        f = np.zeros((x.shape[0], 1))
        g = np.zeros((x.shape[0], 1))
        
        for i in range(x.shape[0]):
            # Calculate stage ratios and payload fraction
            stage_ratios, _ = calculate_stage_ratios(x[i], self.G0, self.ISP, self.EPSILON)
            payload_fraction = np.prod(stage_ratios)
            
            # Store in cache
            self.cache.add(x[i], payload_fraction)
            
            # Store objective (negative since we're minimizing)
            f[i, 0] = -payload_fraction
            
            # Calculate constraint violations using stored config
            g[i, 0] = enforce_stage_constraints(x[i], self.TOTAL_DELTA_V, self.config)
        
        out["F"] = f
        out["G"] = g

def solve_with_slsqp(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Sequential Least Squares Programming (SLSQP)."""
    try:
        logger.debug("Starting SLSQP optimization")
        logger.debug(f"Initial guess: {initial_guess}")
        logger.debug(f"Bounds: {bounds}")
        
        # Ensure arrays
        initial_guess = np.asarray(initial_guess, dtype=float)
        ISP = np.asarray(ISP, dtype=float)
        EPSILON = np.asarray(EPSILON, dtype=float)
        
        def objective(dv):
            return -payload_fraction_objective(dv, G0, ISP, EPSILON)  # Negative since we're minimizing
            
        # Get constraint parameters from config
        constraint_config = config['optimization']['constraints']
        stage_constraints = constraint_config['stage_fractions']
        first_stage_min = stage_constraints['first_stage']['min_fraction']
        first_stage_max = stage_constraints['first_stage']['max_fraction']
        other_stages_min = stage_constraints['other_stages']['min_fraction']
        total_dv_tolerance = constraint_config['total_dv']['tolerance']
        
        def constraint_total_dv(dv):
            # Total delta-v constraint (equality)
            return np.sum(dv) - TOTAL_DELTA_V
            
        def constraint_first_stage_min(dv):
            # First stage minimum (inequality: dv[0] >= min_dv)
            return dv[0] - first_stage_min * TOTAL_DELTA_V
            
        def constraint_first_stage_max(dv):
            # First stage maximum (inequality: dv[0] <= max_dv)
            return first_stage_max * TOTAL_DELTA_V - dv[0]
            
        def constraint_other_stages_min(dv):
            # Other stages minimum (inequality: dv[i] >= min_dv)
            return dv[1:] - other_stages_min * TOTAL_DELTA_V
        
        # Define constraints for scipy's minimize
        constraints = [
            {'type': 'eq', 'fun': constraint_total_dv, 'tol': total_dv_tolerance},
            {'type': 'ineq', 'fun': constraint_first_stage_min},
            {'type': 'ineq', 'fun': constraint_first_stage_max},
        ]
        
        # Add minimum constraints for other stages
        n_stages = len(initial_guess)
        for i in range(1, n_stages):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i: x[i] - other_stages_min * TOTAL_DELTA_V
            })
        
        # Get optimization parameters from config
        opt_config = config.get('optimization', {})
        max_iterations = opt_config.get('max_iterations', 1000)
        
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iterations, 'ftol': total_dv_tolerance}
        )
        
        if not result.success:
            logger.warning(f"SLSQP optimization failed: {result.message}")
            
        # Calculate final stage ratios and payload fraction
        optimal_dv = result.x
        stage_ratios = []
        stages = []
        
        for i, (dv, isp, eps) in enumerate(zip(optimal_dv, ISP, EPSILON)):
            mass_ratio = np.exp(-dv / (G0 * isp))
            lambda_val = mass_ratio - eps  # λᵢ = exp(-ΔVᵢ/(g₀ISPᵢ)) - εᵢ
            stage_ratios.append(lambda_val)
            
            stages.append({
                'stage': i + 1,
                'delta_v': float(dv),
                'Lambda': float(lambda_val),
                'mass_ratio': float(mass_ratio)
            })
            
        payload_fraction = float(np.prod(stage_ratios))
        
        return {
            'success': result.success,
            'message': result.message,
            'payload_fraction': payload_fraction,
            'stages': stages,
            'n_iterations': result.nit,
            'n_function_evals': result.nfev
        }
        
    except Exception as e:
        logger.error(f"Error in SLSQP optimization: {str(e)}")
        raise

def solve_with_basin_hopping(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config, problem=None):
    """Solve using Basin-Hopping.
    
    Args:
        initial_guess: Initial solution vector
        bounds: List of (min, max) bounds for each variable
        G0: Gravitational constant
        ISP: List of specific impulse values
        EPSILON: List of epsilon values
        TOTAL_DELTA_V: Total delta-v constraint
        config: Configuration dictionary
        problem: Optional RocketOptimizationProblem instance. If provided, will use its cache.
    """
    try:
        logger.debug("Starting Basin-Hopping optimization")
        
        # Use provided problem instance or create new one
        if problem is None:
            problem = RocketOptimizationProblem(
                n_var=len(initial_guess),
                bounds=bounds,
                G0=G0,
                ISP=ISP,
                EPSILON=EPSILON,
                TOTAL_DELTA_V=TOTAL_DELTA_V,
                config=config
            )
        
        def objective(dv):
            # Check cache first
            cached_fitness = problem.cache.get_cached_fitness(dv)
            if cached_fitness is not None:
                return -cached_fitness + 1e6 * enforce_stage_constraints(dv, TOTAL_DELTA_V, problem.config)
            
            # Calculate stage ratios and payload fraction
            stage_ratios, _ = calculate_stage_ratios(dv, G0, ISP, EPSILON)
            payload_fraction = np.prod(stage_ratios)
            problem.cache.add(dv, payload_fraction)
            
            # Return negative payload fraction (minimizing) plus constraint penalty
            return -payload_fraction + 1e6 * enforce_stage_constraints(dv, TOTAL_DELTA_V, problem.config)
        
        # Get basin hopping parameters from config
        bh_config = config.get('basin_hopping', {})
        n_iterations = bh_config.get('n_iterations', 100)
        temperature = bh_config.get('temperature', 1.0)
        stepsize = bh_config.get('stepsize', 0.5)
        
        result = basinhopping(
            objective,
            initial_guess,
            minimizer_kwargs={'method': 'SLSQP', 'bounds': bounds},
            niter=n_iterations,
            T=temperature,
            stepsize=stepsize
        )
        
        if not result.lowest_optimization_result.success:
            logger.warning(f"Basin-Hopping optimization failed: {result.message}")
        
        # Calculate final stage ratios and payload fraction
        optimal_dv = result.x
        stage_ratios, mass_ratios = calculate_stage_ratios(
            optimal_dv, G0, ISP, EPSILON
        )
        
        # Build result dictionary
        stages = []
        for i, (dv, mr, sr) in enumerate(zip(optimal_dv, mass_ratios, stage_ratios)):
            stages.append({
                'stage': i + 1,
                'delta_v': float(dv),
                'Lambda': float(sr),
                'mass_ratio': float(mr)
            })
            
        return {
            'success': result.lowest_optimization_result.success,
            'message': str(result.message),
            'payload_fraction': float(np.prod(stage_ratios)),
            'stages': stages,
            'n_iterations': result.nit,
            'n_function_evals': result.nfev
        }
        
    except Exception as e:
        logger.error(f"Error in Basin-Hopping optimization: {str(e)}")
        raise

def solve_with_ga(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config, problem=None):
    """Solve using Genetic Algorithm.
    
    Args:
        initial_guess: Initial solution vector
        bounds: List of (min, max) bounds for each variable
        G0: Gravitational constant
        ISP: List of specific impulse values
        EPSILON: List of epsilon values
        TOTAL_DELTA_V: Total delta-v constraint
        config: Configuration dictionary
        problem: Optional RocketOptimizationProblem instance. If provided, will use its cache.
    """
    try:
        logger.debug("Starting GA optimization")
        
        # Use provided problem instance or create new one
        if problem is None:
            problem = RocketOptimizationProblem(
                n_var=len(initial_guess),
                bounds=bounds,
                G0=G0,
                ISP=ISP,
                EPSILON=EPSILON,
                TOTAL_DELTA_V=TOTAL_DELTA_V,
                config=config
            )
        
        # Get GA parameters from config
        ga_config = config.get('ga', {})
        population_size = ga_config.get('population_size', 100)
        n_generations = ga_config.get('n_generations', 200)
        crossover_prob = ga_config.get('crossover_prob', 0.9)
        crossover_eta = ga_config.get('crossover_eta', 15)
        mutation_prob = ga_config.get('mutation_prob', 0.2)
        mutation_eta = ga_config.get('mutation_eta', 20)
        
        # Create initial population with cached solutions
        initial_population = np.zeros((population_size, len(initial_guess)))
        fitness = np.zeros(initial_population.shape[0])
        
        # Get best solutions from cache
        best_solutions = problem.cache.get_best_solutions()
        n_cached = len(best_solutions)
        n_random = population_size - n_cached - 1
        
        # Start with initial guess
        initial_population[0] = initial_guess
        fitness[0] = problem.cache.get_cached_fitness(initial_guess)
        if fitness[0] is None:
            out = {"F": np.zeros(1), "G": np.zeros(1)}
            problem._evaluate(initial_guess.reshape(1, -1), out)
            fitness[0] = float(-out["F"][0])  # Negate back since _evaluate negates for minimization
            problem.cache.add(initial_guess, fitness[0])
        
        # Add cached solutions
        for i in range(min(n_cached, population_size - 1)):
            initial_population[i + 1] = best_solutions[i]
            fitness[i + 1] = problem.cache.get_cached_fitness(best_solutions[i])
        
        # Fill remaining slots with random solutions
        if n_random > 0:
            random_population = np.random.uniform(
                low=[b[0] for b in bounds],
                high=[b[1] for b in bounds],
                size=(n_random, len(initial_guess))
            )
            initial_population[n_cached + 1:] = random_population
            
            # Evaluate and cache random solutions
            for i in range(n_cached + 1, population_size):
                fitness[i] = problem.cache.get_cached_fitness(initial_population[i])
                if fitness[i] is None:
                    out = {"F": np.zeros(1), "G": np.zeros(1)}
                    problem._evaluate(initial_population[i].reshape(1, -1), out)
                    fitness[i] = float(-out["F"][0])  # Negate back since _evaluate negates for minimization
                    problem.cache.add(initial_population[i], fitness[i])
        
        # Sort population by fitness
        sort_idx = np.argsort(fitness)[::-1]  # Descending order
        initial_population = initial_population[sort_idx]
        fitness = fitness[sort_idx]
        
        # Main GA loop
        algorithm = GA(
            pop_size=population_size,
            sampling=initial_population,
            crossover=SBX(prob=crossover_prob, eta=crossover_eta),
            mutation=PM(prob=mutation_prob, eta=mutation_eta),
            eliminate_duplicates=True
        )
        
        result = pymoo_minimize(
            problem,
            algorithm,
            ('n_gen', n_generations),
            seed=1,
            verbose=False
        )
        
        # Extract best solution
        optimal_dv = result.X
        stage_ratios = []
        stages = []
        
        for i, (dv, isp, eps) in enumerate(zip(optimal_dv, ISP, EPSILON)):
            mass_ratio = np.exp(-dv / (G0 * isp))
            lambda_val = mass_ratio - eps  # λᵢ = exp(-ΔVᵢ/(g₀ISPᵢ)) - εᵢ
            stage_ratios.append(lambda_val)
            
            stages.append({
                'stage': i + 1,
                'delta_v': float(dv),
                'Lambda': float(lambda_val),
                'mass_ratio': float(mass_ratio)
            })
            
        payload_fraction = float(np.prod(stage_ratios))
        
        # Save final solution to cache
        problem.cache.add(optimal_dv, payload_fraction)
        problem.cache.save_cache()
        
        return {
            'success': True,  # GA always completes
            'message': "GA optimization completed",
            'payload_fraction': payload_fraction,
            'stages': stages,
            'n_iterations': result.algorithm.n_gen,
            'n_function_evals': result.algorithm.evaluator.n_eval
        }
        
    except Exception as e:
        logger.error(f"Error in GA optimization: {str(e)}")
        raise

def enforce_stage_constraints(dv_array, total_dv_required, config=None):
    """Enforce stage constraints and return constraint violation value.
    
    Args:
        dv_array: Array of stage delta-v values
        total_dv_required: Required total delta-v
        config: Configuration dictionary containing constraints
        
    Returns:
        float: Constraint violation value (0 if all constraints satisfied)
    """
    if config is None:
        config = {}
    
    # Get constraint parameters from config
    constraints = config.get('constraints', {})
    total_dv_constraint = constraints.get('total_dv', {})
    tolerance = total_dv_constraint.get('tolerance', 1e-6)
    
    # Calculate total delta-v constraint violation
    total_dv = np.sum(dv_array)
    dv_violation = abs(total_dv - total_dv_required)
    
    # Get stage fraction constraints
    stage_fractions = constraints.get('stage_fractions', {})
    first_stage = stage_fractions.get('first_stage', {})
    other_stages = stage_fractions.get('other_stages', {})
    
    # Default constraints if not specified
    min_fraction_first = first_stage.get('min_fraction', 0.15)
    max_fraction_first = first_stage.get('max_fraction', 0.80)
    min_fraction_other = other_stages.get('min_fraction', 0.01)
    max_fraction_other = other_stages.get('max_fraction', 1.0)
    
    # Calculate stage fractions
    stage_fractions = dv_array / total_dv if total_dv > 0 else np.zeros_like(dv_array)
    
    # Check first stage constraints
    if len(stage_fractions) > 0:
        if stage_fractions[0] < min_fraction_first:
            return abs(stage_fractions[0] - min_fraction_first) + dv_violation
        if stage_fractions[0] > max_fraction_first:
            return abs(stage_fractions[0] - max_fraction_first) + dv_violation
    
    # Check other stage constraints
    for fraction in stage_fractions[1:]:
        if fraction < min_fraction_other:
            return abs(fraction - min_fraction_other) + dv_violation
        if fraction > max_fraction_other:
            return abs(fraction - max_fraction_other) + dv_violation
    
    # If we reach here, only return total DV violation
    return dv_violation

def calculate_diversity(population):
    """Calculate population diversity using mean pairwise distance.
    
    Args:
        population: numpy array of shape (pop_size, n_variables)
        
    Returns:
        float: Diversity measure between 0 and 1
    """
    pop_size = population.shape[0]
    if pop_size <= 1:
        return 0.0
        
    # Calculate pairwise distances
    distances = []
    for i in range(pop_size):
        for j in range(i + 1, pop_size):
            dist = np.linalg.norm(population[i] - population[j])
            distances.append(dist)
            
    # Normalize by maximum possible distance
    max_dist = np.linalg.norm(np.ptp(population, axis=0))
    if max_dist == 0:
        return 0.0
        
    mean_dist = np.mean(distances)
    return mean_dist / max_dist

def solve_with_adaptive_ga(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Adaptive Genetic Algorithm with strict constraint handling."""
    try:
        logger.debug("Starting Adaptive GA optimization")
        logger.debug(f"Initial guess: {initial_guess}")
        logger.debug(f"Bounds: {bounds}")
        
        # Create problem instance with caching and config
        problem = RocketOptimizationProblem(
            n_var=len(initial_guess),
            bounds=bounds,
            G0=G0,
            ISP=ISP,
            EPSILON=EPSILON,
            TOTAL_DELTA_V=TOTAL_DELTA_V,
            config=config
        )
        
        # Get Adaptive GA parameters from config
        ada_ga_config = config.get('optimization', {}).get('adaptive_ga', {})
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
        
        # Initialize population
        population = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(initial_pop_size, len(initial_guess))
        )
        # Scale initial population to meet total DV constraint
        for i in range(initial_pop_size):
            total_dv = np.sum(population[i])
            if total_dv > 0:
                population[i] = population[i] * (TOTAL_DELTA_V / total_dv)
        population[0] = initial_guess  # Include initial guess
        population_size = initial_pop_size
        
        # Current parameters
        mutation_rate = initial_mutation_rate
        crossover_rate = initial_crossover_rate
        
        # Best solution tracking
        best_fitness = float('-inf')
        best_solution = None
        stagnation_counter = 0
        
        # Evolution loop
        for generation in range(n_generations):
            # Evaluate population
            out = {"F": np.zeros((population_size, 1)), "G": np.zeros((population_size, 1))}
            problem._evaluate(population, out)
            fitness_values = -out["F"].flatten()
            constraint_violations = out["G"].flatten()
            
            # Apply constraint penalties
            penalty_factor = 1e6
            penalized_fitness = fitness_values - penalty_factor * np.abs(constraint_violations)
            
            # Update best solution only if constraints are satisfied
            feasible_mask = constraint_violations <= 1e-6
            if np.any(feasible_mask):
                feasible_idx = np.where(feasible_mask)[0]
                best_feasible_idx = feasible_idx[np.argmax(penalized_fitness[feasible_idx])]
                if penalized_fitness[best_feasible_idx] > best_fitness:
                    best_fitness = penalized_fitness[best_feasible_idx]
                    best_solution = population[best_feasible_idx].copy()
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
            else:
                stagnation_counter += 1
            
            # Adapt parameters based on progress
            if stagnation_counter >= stagnation_threshold:
                mutation_rate = min(mutation_rate * 1.2, max_mutation_rate)
                population_size = min(int(population_size * 1.1), max_pop_size)
            else:
                mutation_rate = max(mutation_rate * 0.95, min_mutation_rate)
                population_size = max(int(population_size * 0.95), min_pop_size)
            
            # Selection probabilities based on penalized fitness
            selection_probs = penalized_fitness - np.min(penalized_fitness)
            if np.sum(selection_probs) <= 0:
                selection_probs = np.ones_like(penalized_fitness)
            selection_probs = selection_probs / np.sum(selection_probs)
            
            # Select parents
            parent_indices = np.random.choice(
                len(population),
                size=population_size,
                p=selection_probs,
                replace=True
            )
            parents = population[parent_indices]
            
            # Create new population
            new_population = np.zeros((population_size, len(initial_guess)))
            
            # Elitism - preserve best feasible solutions
            elite_size = min(elite_size, population_size)
            if np.any(feasible_mask):
                feasible_sorted = np.argsort(penalized_fitness[feasible_mask])[-elite_size:]
                feasible_indices = np.where(feasible_mask)[0][feasible_sorted]
                new_population[:len(feasible_indices)] = population[feasible_indices]
                remaining_elite = elite_size - len(feasible_indices)
                if remaining_elite > 0:
                    # Fill remaining elite slots with best infeasible solutions
                    infeasible_mask = ~feasible_mask
                    if np.any(infeasible_mask):
                        infeasible_sorted = np.argsort(penalized_fitness[infeasible_mask])[-remaining_elite:]
                        infeasible_indices = np.where(infeasible_mask)[0][infeasible_sorted]
                        new_population[len(feasible_indices):elite_size] = population[infeasible_indices]
            else:
                # If no feasible solutions, use best overall solutions
                elite_indices = np.argsort(penalized_fitness)[-elite_size:]
                new_population[:elite_size] = population[elite_indices]
            
            # Crossover
            for i in range(elite_size, population_size - 1, 2):
                if np.random.random() < crossover_rate:
                    alpha = np.random.random()
                    child1 = alpha * parents[i] + (1 - alpha) * parents[i + 1]
                    child2 = (1 - alpha) * parents[i] + alpha * parents[i + 1]
                    # Scale children to meet total DV constraint
                    total_dv1 = np.sum(child1)
                    total_dv2 = np.sum(child2)
                    if total_dv1 > 0:
                        child1 = child1 * (TOTAL_DELTA_V / total_dv1)
                    if total_dv2 > 0:
                        child2 = child2 * (TOTAL_DELTA_V / total_dv2)
                    new_population[i] = child1
                    new_population[i + 1] = child2
                else:
                    new_population[i] = parents[i]
                    new_population[i + 1] = parents[i + 1]
            
            # Handle last individual if population_size is odd
            if population_size % 2 == 1 and population_size > elite_size:
                new_population[-1] = parents[-1]
            
            # Mutation and constraint enforcement
            for i in range(elite_size, population_size):
                if np.random.random() < mutation_rate:
                    mutation = np.random.normal(0, 0.1, size=len(initial_guess))
                    new_population[i] += mutation
                    # Enforce bounds
                    new_population[i] = np.clip(new_population[i], [b[0] for b in bounds], [b[1] for b in bounds])
                    # Scale to meet total DV constraint
                    total_dv = np.sum(new_population[i])
                    if total_dv > 0:
                        new_population[i] = new_population[i] * (TOTAL_DELTA_V / total_dv)
            
            # Update population
            population = new_population.copy()
            
            # Log progress
            if generation % 10 == 0:
                feasible_count = np.sum(feasible_mask)
                logger.debug(f"Generation {generation}: Best fitness = {best_fitness}")
                logger.debug(f"Feasible solutions: {feasible_count}/{population_size}")
                logger.debug(f"Population size: {population_size}, Mutation rate: {mutation_rate:.3f}")
        
        if best_solution is None:
            logger.warning("No feasible solution found")
            return None
        
        # Calculate final results
        stage_ratios, mass_ratios = calculate_stage_ratios(best_solution, G0, ISP, EPSILON)
        payload_fraction = np.prod(stage_ratios)
        
        # Build stages info
        stages = []
        for i, (dv, mr, sr) in enumerate(zip(best_solution, mass_ratios, stage_ratios)):
            stages.append({
                'stage': i + 1,
                'delta_v': float(dv),
                'Lambda': float(sr),
                'mass_ratio': float(mr)
            })
        
        return {
            'success': True,
            'message': "Adaptive GA optimization completed successfully",
            'payload_fraction': float(payload_fraction),
            'stages': stages,
            'n_iterations': n_generations,
            'n_function_evals': n_generations * initial_pop_size
        }
        
    except Exception as e:
        logger.error(f"Error in optimization: {e}")
        logger.exception("Detailed error information:")
        return None

def solve_with_differential_evolution(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config, problem=None):
    """Solve using Differential Evolution."""
    try:
        logger.debug("Starting Differential Evolution optimization")
        
        # Use provided problem instance or create new one
        if problem is None:
            problem = RocketOptimizationProblem(
                n_var=len(initial_guess),
                bounds=bounds,
                G0=G0,
                ISP=ISP,
                EPSILON=EPSILON,
                TOTAL_DELTA_V=TOTAL_DELTA_V,
                config=config
            )
        
        def objective(dv):
            # Check cache first
            cached_fitness = problem.cache.get_cached_fitness(dv)
            if cached_fitness is not None:
                return -cached_fitness
            
            # Calculate stage ratios and payload fraction
            stage_ratios, _ = calculate_stage_ratios(dv, G0, ISP, EPSILON)
            payload_fraction = np.prod(stage_ratios)
            problem.cache.add(dv, payload_fraction)
            
            # Return negative payload fraction (minimizing)
            return -payload_fraction
            
        # Get stage constraints from config
        constraint_config = config['optimization']['constraints']
        stage_constraints = constraint_config['stage_fractions']
        first_stage_min = stage_constraints['first_stage']['min_fraction']
        first_stage_max = stage_constraints['first_stage']['max_fraction']
        other_stages_min = stage_constraints['other_stages']['min_fraction']
        total_dv_tolerance = constraint_config['total_dv']['tolerance']
        
        def constraint_total_dv(dv):
            # Total delta-v constraint (equality)
            return np.sum(dv) - TOTAL_DELTA_V
            
        def constraint_first_stage_min(dv):
            # First stage minimum (inequality: dv[0] >= min_dv)
            return dv[0] - first_stage_min * TOTAL_DELTA_V
            
        def constraint_first_stage_max(dv):
            # First stage maximum (inequality: dv[0] <= max_dv)
            return first_stage_max * TOTAL_DELTA_V - dv[0]
            
        def constraint_other_stages_min(dv):
            # Other stages minimum (inequality: dv[i] >= min_dv)
            return dv[1:] - other_stages_min * TOTAL_DELTA_V
        
        # Define constraints for scipy's differential_evolution
        constraints = [
            {'type': 'eq', 'fun': constraint_total_dv, 'tol': total_dv_tolerance},
            {'type': 'ineq', 'fun': constraint_first_stage_min},
            {'type': 'ineq', 'fun': constraint_first_stage_max},
        ]
        
        # Add minimum constraints for other stages
        n_stages = len(initial_guess)
        for i in range(1, n_stages):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i: x[i] - other_stages_min * TOTAL_DELTA_V
            })
        
        # Get DE parameters from config
        de_config = config.get('differential_evolution', {})
        population_size = de_config.get('population_size', 15)
        max_iter = de_config.get('max_iterations', 1000)
        mutation = de_config.get('mutation', [0.5, 1.0])
        recombination = de_config.get('recombination', 0.7)
        strategy = de_config.get('strategy', 'best1bin')
        tol = de_config.get('tol', 1e-6)
        
        result = differential_evolution(
            objective,
            bounds,
            strategy=strategy,
            maxiter=max_iter,
            popsize=population_size,
            tol=tol,
            mutation=mutation,
            recombination=recombination,
            init='sobol',
            constraints=constraints
        )
        
        if not result.success:
            logger.warning(f"Differential Evolution optimization failed: {result.message}")
        
        # Calculate final stage ratios and payload fraction
        optimal_dv = result.x
        stage_ratios, mass_ratios = calculate_stage_ratios(
            optimal_dv, G0, ISP, EPSILON
        )
        
        # Build result dictionary
        stages = []
        for i, (dv, mr, sr) in enumerate(zip(optimal_dv, mass_ratios, stage_ratios)):
            stages.append({
                'stage': i + 1,
                'delta_v': float(dv),
                'Lambda': float(sr),
                'mass_ratio': float(mr)
            })
        
        return {
            'success': result.success,
            'message': str(result.message),
            'payload_fraction': float(np.prod(stage_ratios)),
            'stages': stages,
            'n_iterations': result.nit,
            'n_function_evals': result.nfev
        }
        
    except Exception as e:
        logger.error(f"Error in Differential Evolution optimization: {str(e)}")
        raise

def solve_with_pso(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Particle Swarm Optimization."""
    try:
        logger.debug("Starting PSO optimization")
        
        # Ensure arrays
        initial_guess = np.asarray(initial_guess, dtype=float)
        ISP = np.asarray(ISP, dtype=float)
        EPSILON = np.asarray(EPSILON, dtype=float)
        
        def objective(dv):
            # Calculate fitness with penalty for constraint violations
            penalty = enforce_stage_constraints(dv, TOTAL_DELTA_V, config)
            if penalty > 0:
                return float('-inf')  # Invalid solution
            return payload_fraction_objective(dv, G0, ISP, EPSILON)
        
        # Get PSO parameters from config with defaults
        pso_config = config.get('optimization', {}).get('pso', {})
        n_particles = pso_config.get('n_particles', 50)
        n_iterations = pso_config.get('n_iterations', 200)
        w = pso_config.get('inertia_weight', 0.7)
        c1 = pso_config.get('cognitive_param', 1.5)
        c2 = pso_config.get('social_param', 1.5)
        
        # Initialize particles
        n_var = len(initial_guess)
        particles = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(n_particles, n_var)
        )
        particles[0] = initial_guess  # Use initial guess as first particle
        
        # Initialize velocities
        v_max = 0.1 * (TOTAL_DELTA_V / n_var)
        velocities = np.random.uniform(-v_max, v_max, size=(n_particles, n_var))
        
        # Initialize best positions and values
        pbest = particles.copy()
        pbest_values = np.array([objective(p) for p in particles])
        gbest = pbest[np.argmax(pbest_values)]  # Using argmax since we're maximizing
        gbest_value = np.max(pbest_values)
        
        # Main PSO loop
        for _ in range(n_iterations):
            # Update velocities
            r1, r2 = np.random.rand(2)
            velocities = (w * velocities +
                        c1 * r1 * (pbest - particles) +
                        c2 * r2 * (gbest - particles))
            velocities = np.clip(velocities, -v_max, v_max)
            
            # Update positions
            particles += velocities
            particles = np.clip(particles, [b[0] for b in bounds], [b[1] for b in bounds])
            
            # Update personal bests
            values = np.array([objective(p) for p in particles])
            improved = values > pbest_values
            pbest[improved] = particles[improved]
            pbest_values[improved] = values[improved]
            
            # Update global best
            max_idx = np.argmax(values)
            if values[max_idx] > gbest_value:
                gbest = particles[max_idx].copy()
                gbest_value = values[max_idx]
        
        # Calculate final stage ratios and payload fraction
        stage_ratios, mass_ratios = calculate_stage_ratios(gbest, G0, ISP, EPSILON)
        payload_fraction = np.prod(stage_ratios)
        
        # Check if solution is valid
        penalty = enforce_stage_constraints(gbest, TOTAL_DELTA_V, config)
        is_valid = penalty == 0
        
        # Build result dictionary
        stages = []
        for i, (dv, mr, sr) in enumerate(zip(gbest, mass_ratios, stage_ratios)):
            stages.append({
                'stage': i + 1,
                'delta_v': float(dv),
                'Lambda': float(sr),
                'mass_ratio': float(mr)
            })
        
        return {
            'success': is_valid,
            'message': "PSO optimization completed" if is_valid else "PSO failed to find valid solution",
            'payload_fraction': float(payload_fraction),
            'stages': stages,
            'n_iterations': n_iterations,
            'n_function_evals': n_iterations * n_particles
        }
        
    except Exception as e:
        logger.error(f"Error in PSO optimization: {str(e)}")
        raise
