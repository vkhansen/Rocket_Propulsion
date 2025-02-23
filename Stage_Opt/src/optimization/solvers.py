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

        algorithm = GA(
            pop_size=config["optimization"]["population_size"],
            eliminate_duplicates=True,
            mutation=PolynomialMutation(prob=0.2, eta=20),
            crossover=SBX(prob=0.9, eta=15),
            repair=RepairDeltaV()
        )

        termination = DefaultSingleObjectiveTermination(
            xtol=1e-6,
            cvtol=1e-6,
            ftol=1e-6,
            period=20,
            n_max_gen=config["optimization"]["max_iterations"],
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
        result = differential_evolution(
            objective,
            bounds,
            strategy='best1bin',
            maxiter=config["optimization"]["max_iterations"],
            popsize=config["optimization"]["population_size"],
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.7,
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
    start_time = time.time()
    
    try:
        logger.info(f"Starting Adaptive GA optimization with parameters:")
        logger.info(f"Initial guess: {initial_guess}")
        logger.info(f"G0: {G0}, ISP: {ISP}, EPSILON: {EPSILON}")
        logger.info(f"TOTAL_DELTA_V: {TOTAL_DELTA_V}")
        
        # Initialize adaptive GA
        ga_config = config["optimization"]["adaptive_ga"]
        aga = AdaptiveGA(
            config=config,
            n_vars=len(initial_guess),
            bounds=bounds,
            total_delta_v=TOTAL_DELTA_V,
            isp=ISP,
            epsilon=EPSILON
        )
        
        # Run optimization
        result = aga.optimize()
        
        if result is None:
            logger.error("Adaptive GA optimization failed")
            return None
            
        # Calculate performance metrics
        execution_time = time.time() - start_time
        optimal_dv = result['optimal_dv']
        stage_ratios = calculate_mass_ratios(optimal_dv, ISP, EPSILON, G0)
        payload_fraction = calculate_payload_fraction(stage_ratios)
        
        logger.info(f"Adaptive GA optimization succeeded:")
        logger.info(f"  Delta-V: {[f'{dv:.2f}' for dv in optimal_dv]} m/s")
        logger.info(f"  Mass ratios: {[f'{r:.3f}' for r in stage_ratios]}")
        logger.info(f"  Payload fraction: {payload_fraction:.3f}")
        
        return {
            'method': 'Adaptive GA',
            'optimal_dv': optimal_dv,
            'stage_ratios': stage_ratios,
            'payload_fraction': float(payload_fraction),
            'execution_time': execution_time,
            'history': result['history']
        }
        
    except Exception as e:
        logger.error(f"Error in Adaptive GA optimization: {e}")
        return None

class AdaptiveGA:
    def __init__(self, config, n_vars, bounds, total_delta_v, isp, epsilon):
        """Initialize the Adaptive GA."""
        self.config = config["optimization"]["adaptive_ga"]
        self.n_vars = n_vars
        self.bounds = np.array(bounds)  # Convert bounds to numpy array
        self.bounds_low = np.array([b[0] for b in bounds])
        self.bounds_high = np.array([b[1] for b in bounds])
        self.pop_size = self.config["initial_pop_size"]
        self.mutation_rate = self.config["initial_mutation_rate"]
        self.crossover_rate = self.config["initial_crossover_rate"]
        self.best_fitness_history = []
        self.diversity_history = []
        self.stagnation_counter = 0
        self.total_delta_v = total_delta_v
        self.ISP = isp
        self.EPSILON = epsilon
        self.history = []  # Store optimization history
        self.execution_time = 0  # Track execution time
        
    def initialize_population(self):
        """Initialize population with smart initialization strategies."""
        population = []
        n_equal = self.pop_size // 3
        
        # Strategy 1: Equal distribution
        for _ in range(n_equal):
            sol = np.full(self.n_vars, self.total_delta_v / self.n_vars)
            sol += np.random.normal(0, self.total_delta_v * 0.05, self.n_vars)
            sol = np.clip(sol, self.bounds_low, self.bounds_high)
            sol = sol * (self.total_delta_v / np.sum(sol))
            population.append(sol)
            
        # Strategy 2: ISP-weighted distribution
        n_isp = (self.pop_size - n_equal) // 2
        for _ in range(n_isp):
            weights = np.array(self.ISP) / np.sum(self.ISP)
            sol = weights * self.total_delta_v
            sol += np.random.normal(0, self.total_delta_v * 0.05, self.n_vars)
            sol = np.clip(sol, self.bounds_low, self.bounds_high)
            sol = sol * (self.total_delta_v / np.sum(sol))
            population.append(sol)
            
        # Strategy 3: Random distribution
        while len(population) < self.pop_size:
            sol = np.random.uniform(self.bounds_low, self.bounds_high, self.n_vars)
            sol = sol * (self.total_delta_v / np.sum(sol))
            population.append(sol)
            
        return np.array(population)

    def evaluate_fitness(self, individual):
        """Evaluate the fitness of an individual."""
        try:
            individual = np.asarray(individual).flatten()
            
            # Check total delta-v constraint with a softer penalty
            delta_v_error = abs(np.sum(individual) - self.total_delta_v)
            if delta_v_error > 1e-4:  # Relaxed tolerance
                penalty = 100.0 * delta_v_error
            else:
                penalty = 0.0
                
            # Check bounds constraints with a softer penalty
            bounds_violation = np.sum(np.maximum(0, self.bounds_low - individual)) + \
                             np.sum(np.maximum(0, individual - self.bounds_high))
            if bounds_violation > 0:
                penalty += 100.0 * bounds_violation
                
            # Calculate payload fraction
            mass_ratios = calculate_mass_ratios(individual, self.ISP, self.EPSILON)
            if mass_ratios is None:
                return -np.inf
                
            payload_fraction = calculate_payload_fraction(mass_ratios)
            
            # Return fitness with penalty
            return float(payload_fraction) - penalty
            
        except Exception as e:
            logger.error(f"Error in fitness evaluation: {e}")
            return -np.inf

    def mutation(self, individual):
        """Perform adaptive mutation with repair."""
        try:
            if np.random.random() > self.mutation_rate:
                return individual.copy()
                
            # Calculate adaptive mutation strength based on bounds and current fitness
            mutation_strength = (self.bounds_high - self.bounds_low) * 0.1 * \
                              (1.0 - len(self.history) / self.config["n_generations"])
            
            # Apply mutation with varying strength per dimension
            mutation = np.random.normal(0, mutation_strength, self.n_vars)
            
            # Apply mutation and repair
            mutated = individual + mutation
            
            # Repair bounds violations while maintaining proportions
            if np.any(mutated < self.bounds_low) or np.any(mutated > self.bounds_high):
                # Clip to bounds
                mutated = np.clip(mutated, self.bounds_low, self.bounds_high)
                
            # Repair total delta-v constraint while preserving relative proportions
            current_sum = np.sum(mutated)
            if abs(current_sum - self.total_delta_v) > 1e-6:
                scale = self.total_delta_v / current_sum
                mutated = mutated * scale
                
                # If scaling caused bounds violations, iteratively repair
                iteration = 0
                while (np.any(mutated < self.bounds_low) or \
                       np.any(mutated > self.bounds_high)) and \
                       iteration < 10:
                    # Clip to bounds
                    mutated = np.clip(mutated, self.bounds_low, self.bounds_high)
                    # Rescale to meet total delta-v
                    current_sum = np.sum(mutated)
                    if abs(current_sum - self.total_delta_v) > 1e-6:
                        scale = self.total_delta_v / current_sum
                        mutated = mutated * scale
                    iteration += 1
            
            return mutated
            
        except Exception as e:
            logger.error(f"Error in mutation: {e}")
            return individual.copy()

    def selection(self, population, fitnesses, tournament_size=3):
        """Tournament selection with elitism."""
        try:
            population = np.asarray(population)
            fitnesses = np.asarray(fitnesses).flatten()
            
            # Keep the best solution (elitism)
            elite_idx = np.argmax(fitnesses)
            elite = population[elite_idx].copy()
            selected = [elite]
            
            # Tournament selection for the rest
            while len(selected) < self.pop_size:
                tournament_idx = np.random.choice(len(population), tournament_size, replace=False)
                tournament_fitnesses = fitnesses[tournament_idx]
                winner_idx = tournament_idx[np.argmax(tournament_fitnesses)]
                selected.append(population[winner_idx].copy())
                
            return np.array(selected)
            
        except Exception as e:
            logger.error(f"Error in selection: {e}")
            return population  # Return original population if error occurs

    def crossover(self, parent1, parent2):
        """Perform arithmetic crossover with repair."""
        try:
            if np.random.random() > self.crossover_rate:
                return parent1.copy(), parent2.copy()
                
            # Calculate adaptive crossover strength based on bounds and current fitness
            crossover_strength = (self.bounds_high - self.bounds_low) * 0.1 * \
                              (1.0 - len(self.history) / self.config["n_generations"])
            
            # Apply crossover with varying strength per dimension
            alpha = np.random.rand(self.n_vars)
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = (1 - alpha) * parent1 + alpha * parent2
            
            # Repair to maintain total delta-v and bounds
            for child in [child1, child2]:
                # Clip to bounds
                child = np.clip(child, self.bounds_low, self.bounds_high)
                # Rescale to meet total delta-v
                current_sum = np.sum(child)
                if abs(current_sum - self.total_delta_v) > 1e-6:
                    scale_factor = self.total_delta_v / current_sum
                    child *= scale_factor
                    # If scaling caused bounds violations, iteratively repair
                    iteration = 0
                    while (np.any(child < self.bounds_low) or \
                           np.any(child > self.bounds_high)) and \
                           iteration < 10:
                        child = np.clip(child, self.bounds_low, self.bounds_high)
                        current_sum = np.sum(child)
                        if abs(current_sum - self.total_delta_v) > 1e-6:
                            scale_factor = self.total_delta_v / current_sum
                            child *= scale_factor
                        iteration += 1
            
            return child1, child2
            
        except Exception as e:
            logger.error(f"Error in crossover: {e}")
            return parent1.copy(), parent2.copy()

    def calculate_diversity(self, population):
        """Calculate population diversity using multiple metrics."""
        try:
            # Standard deviation across dimensions
            std_diversity = np.mean(np.std(population, axis=0))
            
            # Average pairwise Euclidean distance
            n_samples = min(len(population), 10)  # Limit computation for large populations
            if n_samples < len(population):
                idx = np.random.choice(len(population), n_samples, replace=False)
                sample_pop = population[idx]
            else:
                sample_pop = population
                
            distances = []
            for i in range(len(sample_pop)):
                for j in range(i + 1, len(sample_pop)):
                    dist = np.linalg.norm(sample_pop[i] - sample_pop[j])
                    distances.append(dist)
            
            if distances:
                distance_diversity = np.mean(distances)
            else:
                distance_diversity = 0.0
                
            # Combine metrics
            return (std_diversity + distance_diversity) / 2
            
        except Exception as e:
            logger.error(f"Error in diversity calculation: {e}")
            return 0.0

    def update_parameters(self, population, fitnesses, generations_without_improvement):
        """Update adaptive parameters based on current state."""
        diversity = self.calculate_diversity(population)
        mean_fitness = np.mean(fitnesses)
        best_fitness = np.max(fitnesses)
        
        # Record metrics
        self.history.append({
            'generation': len(self.history),
            'best_fitness': best_fitness,
            'mean_fitness': mean_fitness,
            'diversity': diversity,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'population_size': self.pop_size
        })
        
        # Adjust mutation rate based on diversity
        if diversity < self.config["diversity_threshold"]:
            self.mutation_rate = min(self.mutation_rate * 1.5, self.config["max_mutation_rate"])
        else:
            self.mutation_rate = max(self.mutation_rate * 0.9, self.config["min_mutation_rate"])
        
        # Adjust crossover rate based on fitness improvement
        if generations_without_improvement > 5:
            self.crossover_rate = min(self.crossover_rate * 1.1, self.config["max_crossover_rate"])
        else:
            self.crossover_rate = max(self.crossover_rate * 0.9, self.config["min_crossover_rate"])

    def optimize(self):
        """Run the optimization process."""
        try:
            start_time = time.time()
            population = self.initialize_population()
            best_fitness = -np.inf
            best_solution = None
            generations_without_improvement = 0
            
            # Track best solution history
            best_history = []
            
            for generation in range(self.config["n_generations"]):
                # Evaluate population
                fitnesses = np.array([self.evaluate_fitness(ind) for ind in population])
                valid_mask = np.isfinite(fitnesses)
                
                if not np.any(valid_mask):
                    logger.warning(f"No valid solutions in generation {generation}")
                    continue
                
                current_best_fitness = np.max(fitnesses[valid_mask])
                current_best_idx = np.argmax(fitnesses)
                
                # Update best solution
                if current_best_fitness > best_fitness:
                    improvement = current_best_fitness - best_fitness
                    best_fitness = current_best_fitness
                    best_solution = population[current_best_idx].copy()
                    generations_without_improvement = 0
                    best_history.append((generation, best_fitness, best_solution.copy()))
                    
                    logger.info(f"Generation {generation}: New best solution found")
                    logger.info(f"Fitness improvement: {improvement:.6f}")
                    logger.info(f"Current best fitness: {best_fitness:.6f}")
                else:
                    generations_without_improvement += 1
                
                # Check for convergence
                if generations_without_improvement >= self.config["stagnation_threshold"]:
                    logger.info(f"Optimization converged after {generation} generations")
                    break
                
                # Evolution
                selected = self.selection(population, fitnesses)
                new_population = []
                
                # Ensure we keep the best solution (elitism)
                new_population.append(best_solution.copy())
                
                # Crossover and mutation for the rest of the population
                while len(new_population) < self.pop_size:
                    idx1, idx2 = np.random.choice(len(selected), 2, replace=False)
                    parent1, parent2 = selected[idx1], selected[idx2]
                    child1, child2 = self.crossover(parent1, parent2)
                    child1 = self.mutation(child1)
                    child2 = self.mutation(child2)
                    new_population.extend([child1, child2])
                
                # Trim population to exact size if needed
                if len(new_population) > self.pop_size:
                    new_population = new_population[:self.pop_size]
                
                population = np.array(new_population)
                self.update_parameters(population, fitnesses, generations_without_improvement)
            
            self.execution_time = time.time() - start_time
            
            if best_solution is not None:
                # Calculate final metrics
                mass_ratios = calculate_mass_ratios(best_solution, self.ISP, self.EPSILON)
                if mass_ratios is not None:
                    payload_fraction = calculate_payload_fraction(mass_ratios)
                    
                    # Add convergence history to results
                    convergence_history = [(gen, fit) for gen, fit, _ in best_history]
                    
                    return {
                        'method': 'GA-Adaptive',
                        'optimal_dv': best_solution.tolist(),
                        'stage_ratios': mass_ratios.tolist(),
                        'payload_fraction': float(payload_fraction),
                        'execution_time': self.execution_time,
                        'history': self.history,
                        'convergence_history': convergence_history
                    }
            
            logger.warning("Optimization failed to find a valid solution")
            return None
            
        except Exception as e:
            logger.error(f"Error in optimization: {e}")
            return None

def solve_with_pso(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Particle Swarm Optimization."""
    start_time = time.time()
    
    try:
        logger.info(f"Starting PSO optimization with parameters:")
        logger.info(f"Initial guess: {initial_guess}")
        logger.info(f"G0: {G0}, ISP: {ISP}, EPSILON: {EPSILON}")
        logger.info(f"TOTAL_DELTA_V: {TOTAL_DELTA_V}")
        
        pso_config = config.get("optimization", {}).get("pso", {})
        n_particles = pso_config.get("n_particles", 50)
        max_iterations = pso_config.get("max_iterations", 100)
        
        # Initialize particles
        particles = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(n_particles, len(initial_guess))
        )
        
        # Scale initial particles to satisfy total delta-V constraint
        sums = np.sum(particles, axis=1)
        non_zero_mask = sums > 0  # Avoid division by zero
        if np.any(non_zero_mask):
            # Scale to meet total delta-V
            scale_factors = np.where(non_zero_mask, TOTAL_DELTA_V / sums, 0)
            particles[non_zero_mask] *= scale_factors[non_zero_mask, np.newaxis]
            
        velocities = np.zeros_like(particles)
        personal_best_pos = particles.copy()
        personal_best_val = np.array([payload_fraction_objective(p, G0, ISP, EPSILON) for p in particles])
        global_best_pos = personal_best_pos[np.argmin(personal_best_val)]
        global_best_val = np.min(personal_best_val)
        
        w = pso_config.get("inertia", 0.7)
        c1 = pso_config.get("cognitive", 1.5)
        c2 = pso_config.get("social", 1.5)
        
        for iteration in range(max_iterations):
            # Update velocities and positions
            r1, r2 = np.random.rand(2, n_particles, len(initial_guess))
            velocities = (w * velocities +
                        c1 * r1 * (personal_best_pos - particles) +
                        c2 * r2 * (global_best_pos - particles))
            
            particles += velocities
            
            # Clip to bounds
            for j in range(len(initial_guess)):
                particles[:, j] = np.clip(particles[:, j], bounds[j][0], bounds[j][1])
            
            # Scale to meet total delta-V constraint
            sums = np.sum(particles, axis=1)
            non_zero_mask = sums > 0
            if np.any(non_zero_mask):
                scale_factors = np.where(non_zero_mask, TOTAL_DELTA_V / sums, 0)
                scale_factors = scale_factors.reshape(-1, 1)  # Reshape for broadcasting
                particles[non_zero_mask] *= scale_factors[non_zero_mask]
            
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
                
        # Calculate final results
        execution_time = time.time() - start_time
        optimal_dv = list(map(float, global_best_pos))  # Convert to list of floats
        stage_ratios = calculate_mass_ratios(optimal_dv, ISP, EPSILON, G0)
        payload_fraction = calculate_payload_fraction(stage_ratios)
        
        logger.info(f"PSO optimization succeeded:")
        logger.info(f"  Delta-V: {[f'{dv:.2f}' for dv in optimal_dv]} m/s")
        logger.info(f"  Mass ratios: {[f'{r:.3f}' for r in stage_ratios]}")
        logger.info(f"  Payload fraction: {payload_fraction:.3f}")
        
        return {
            'method': 'PSO',
            'optimal_dv': optimal_dv,
            'stage_ratios': list(map(float, stage_ratios)),
            'payload_fraction': float(payload_fraction),
            'execution_time': float(execution_time)
        }
        
    except Exception as e:
        logger.error(f"Error in PSO optimization: {e}")
        return None
