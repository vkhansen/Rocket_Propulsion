"""Optimization solvers."""
import numpy as np
from scipy.optimize import minimize, basinhopping, differential_evolution
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.core.repair import Repair
from pymoo.termination.default import DefaultSingleObjectiveTermination
import time

from .objective import payload_fraction_objective, objective_with_penalty
from ..utils.config import logger
from ..utils.data import calculate_mass_ratios, calculate_payload_fraction

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
        logger.info(f"Starting SLSQP optimization with parameters:")
        logger.info(f"Initial guess: {initial_guess}")
        logger.info(f"G0: {G0}, ISP: {ISP}, EPSILON: {EPSILON}")
        logger.info(f"TOTAL_DELTA_V: {TOTAL_DELTA_V}")
        
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
        else:
            # Calculate performance metrics
            mass_ratios = calculate_mass_ratios(result.x, ISP, EPSILON, G0)
            payload_fraction = calculate_payload_fraction(mass_ratios)
            logger.info(f"SLSQP optimization succeeded:")
            logger.info(f"  Delta-V: {[f'{dv:.2f}' for dv in result.x]} m/s")
            logger.info(f"  Mass ratios: {[f'{r:.3f}' for r in mass_ratios]}")
            logger.info(f"  Payload fraction: {payload_fraction:.3f}")
            
        return result.x
        
    except Exception as e:
        logger.error(f"SLSQP optimization failed: {e}")
        raise

def solve_with_basin_hopping(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Basin-Hopping."""
    try:
        logger.info(f"Starting Basin-Hopping optimization with parameters:")
        logger.info(f"Initial guess: {initial_guess}")
        logger.info(f"G0: {G0}, ISP: {ISP}, EPSILON: {EPSILON}")
        logger.info(f"TOTAL_DELTA_V: {TOTAL_DELTA_V}")
        
        def objective(dv):
            return objective_with_penalty(dv, G0, ISP, EPSILON, TOTAL_DELTA_V)
            
        minimizer_kwargs = {
            "method": "SLSQP",
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
            niter=config["optimization"]["basin_hopping"]["n_iterations"],
            T=config["optimization"]["basin_hopping"]["temperature"],
            stepsize=config["optimization"]["basin_hopping"]["step_size"]
        )
        
        if not result.lowest_optimization_result.success:
            logger.warning(f"Basin-Hopping optimization warning: {result.message}")
        else:
            # Calculate performance metrics
            mass_ratios = calculate_mass_ratios(result.x, ISP, EPSILON, G0)
            payload_fraction = calculate_payload_fraction(mass_ratios)
            logger.info(f"Basin-Hopping optimization succeeded:")
            logger.info(f"  Delta-V: {[f'{dv:.2f}' for dv in result.x]} m/s")
            logger.info(f"  Mass ratios: {[f'{r:.3f}' for r in mass_ratios]}")
            logger.info(f"  Payload fraction: {payload_fraction:.3f}")
            
        return result.x
        
    except Exception as e:
        logger.error(f"Basin-Hopping optimization failed: {e}")
        raise

def solve_with_differential_evolution(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Differential Evolution."""
    try:
        logger.info(f"Starting Differential Evolution optimization with parameters:")
        logger.info(f"Initial guess: {initial_guess}")
        logger.info(f"G0: {G0}, ISP: {ISP}, EPSILON: {EPSILON}")
        logger.info(f"TOTAL_DELTA_V: {TOTAL_DELTA_V}")
        
        def objective(dv):
            return objective_with_penalty(dv, G0, ISP, EPSILON, TOTAL_DELTA_V)
            
        result = differential_evolution(
            objective,
            bounds,
            strategy=config["optimization"]["differential_evolution"]["strategy"],
            maxiter=config["optimization"]["differential_evolution"]["max_iterations"],
            popsize=config["optimization"]["differential_evolution"]["population_size"],
            tol=config["optimization"]["differential_evolution"]["tol"],
            mutation=config["optimization"]["differential_evolution"]["mutation"],
            recombination=config["optimization"]["differential_evolution"]["recombination"]
        )
        
        if not result.success:
            logger.warning(f"Differential Evolution optimization warning: {result.message}")
        else:
            # Calculate performance metrics
            mass_ratios = calculate_mass_ratios(result.x, ISP, EPSILON, G0)
            payload_fraction = calculate_payload_fraction(mass_ratios)
            logger.info(f"Differential Evolution optimization succeeded:")
            logger.info(f"  Delta-V: {[f'{dv:.2f}' for dv in result.x]} m/s")
            logger.info(f"  Mass ratios: {[f'{r:.3f}' for r in mass_ratios]}")
            logger.info(f"  Payload fraction: {payload_fraction:.3f}")
            
        return result.x
        
    except Exception as e:
        logger.error(f"Differential Evolution optimization failed: {e}")
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
                        scale = TOTAL_DELTA_V / current_sum
                        x = x * scale
                        # Ensure bounds are satisfied while maintaining proportions
                        x_clipped = np.clip(x, problem.xl, problem.xu)
                        if not np.array_equal(x, x_clipped):
                            # If clipping changed values, rescale while preserving proportions
                            x = x_clipped
                            current_sum = np.sum(x)
                            if abs(current_sum - TOTAL_DELTA_V) > 1e-6:
                                x = x * (TOTAL_DELTA_V / current_sum)
                        X[i] = x
                return X

        class OptimizationProblem(Problem):
            """Problem class for GA optimization."""
            def __init__(self, n_var, n_obj, xl, xu):
                super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)
                self.total_delta_v = None
                self.G0 = None
                self.ISP = None
                self.EPSILON = None

            def _evaluate(self, x, out, *args, **kwargs):
                """Evaluate the objective function for GA."""
                f = []
                for dv in x:
                    # Calculate payload fraction
                    stage_ratios = [np.exp(-dvi / (self.G0 * isp)) - eps 
                                  for dvi, isp, eps in zip(dv, self.ISP, self.EPSILON)]
                    payload = -np.prod(stage_ratios)  # Negative because GA minimizes
                    
                    # Add softer penalty for total ΔV constraint
                    delta_v_error = abs(np.sum(dv) - self.total_delta_v)
                    penalty = 100.0 * delta_v_error if delta_v_error > 1e-6 else 0.0
                    f.append(payload + penalty)
                    
                out["F"] = np.column_stack([f])

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
            pop_size=config["optimization"]["ga"]["population_size"],
            eliminate_duplicates=True,
            mutation=PolynomialMutation(
                prob=config["optimization"]["ga"]["mutation_prob"],
                eta=config["optimization"]["ga"]["mutation_eta"]
            ),
            crossover=SBX(
                prob=config["optimization"]["ga"]["crossover_prob"],
                eta=config["optimization"]["ga"]["crossover_eta"]
            ),
            repair=RepairDeltaV()
        )

        termination = DefaultSingleObjectiveTermination(
            xtol=1e-4,  # Relaxed tolerance
            cvtol=1e-4,  # Relaxed tolerance
            ftol=1e-4,  # Relaxed tolerance
            period=20,
            n_max_gen=config["optimization"]["ga"]["n_generations"],
            n_max_evals=None
        )
        
        res = pymoo_minimize(
            problem,
            algorithm,
            termination,
            seed=1,
            verbose=True  # Enable verbose output for debugging
        )

        if res.X is None:
            logger.warning(f"GA optimization failed to find a solution: {res.message}")
            return None
            
        # Verify the solution meets the total delta-v constraint
        if abs(np.sum(res.X) - TOTAL_DELTA_V) > 1e-4:
            logger.warning(f"GA solution violates total delta-v constraint by {abs(np.sum(res.X) - TOTAL_DELTA_V)}")
            return None
            
        return res.X
        
    except Exception as e:
        logger.error(f"GA optimization failed with error: {str(e)}")
        return None

def solve_with_adaptive_ga(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Adaptive Genetic Algorithm."""
    try:
        logger.info(f"Starting Adaptive GA optimization with parameters:")
        logger.info(f"Initial guess: {initial_guess}")
        logger.info(f"G0: {G0}, ISP: {ISP}, EPSILON: {EPSILON}")
        logger.info(f"TOTAL_DELTA_V: {TOTAL_DELTA_V}")
        
        n_vars = len(initial_guess)
        
        # Initialize and run the adaptive GA
        ga = AdaptiveGA(config, n_vars, bounds, TOTAL_DELTA_V, ISP, EPSILON)
        result = ga.optimize()
        
        if result is not None:
            logger.info(f"Adaptive GA optimization succeeded:")
            logger.info(f"  Delta-V: {[f'{dv:.2f}' for dv in result['optimal_dv']]} m/s")
            logger.info(f"  Mass ratios: {[f'{r:.3f}' for r in result['stage_ratios']]}")
            logger.info(f"  Payload fraction: {result['payload_fraction']:.3f}")
            logger.info(f"  Time: {result['execution_time']:.3f} seconds")
        
        return result
        
    except Exception as e:
        logger.error(f"Adaptive GA optimization failed: {e}")
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
                
            # Arithmetic crossover with random weights per dimension
            alpha = np.random.random(self.n_vars)
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = (1 - alpha) * parent1 + alpha * parent2
            
            # Repair to maintain total delta-v and bounds
            for child in [child1, child2]:
                # Clip to bounds
                child = np.clip(child, self.bounds_low, self.bounds_high)
                # Rescale to meet total delta-v
                current_sum = np.sum(child)
                if abs(current_sum - self.total_delta_v) > 1e-6:
                    scale = self.total_delta_v / current_sum
                    child = child * scale
                    # If scaling caused bounds violations, iteratively repair
                    iteration = 0
                    while (np.any(child < self.bounds_low) or \
                           np.any(child > self.bounds_high)) and \
                           iteration < 10:
                        child = np.clip(child, self.bounds_low, self.bounds_high)
                        current_sum = np.sum(child)
                        if abs(current_sum - self.total_delta_v) > 1e-6:
                            scale = self.total_delta_v / current_sum
                            child = child * scale
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
    """Solve using Particle Swarm Optimization (PSO).
    
    Args:
        initial_guess (list): Initial solution guess
        bounds (list): List of tuples defining bounds for each variable
        G0 (float): Gravitational constant
        ISP (list): List of specific impulses for each stage
        EPSILON (list): List of structural coefficients for each stage
        TOTAL_DELTA_V (float): Total required delta-V
        config (dict): Configuration parameters
        
    Returns:
        dict: Dictionary containing optimization results
    """
    start_time = time.time()
    
    try:
        n_particles = config.get('pso_particles', 50)
        n_iterations = config.get('pso_iterations', 100)
        c1 = config.get('pso_c1', 2.0)  # Cognitive parameter
        c2 = config.get('pso_c2', 2.0)  # Social parameter
        w = config.get('pso_w', 0.7)     # Inertia weight
        
        # Initialize particles
        particles = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(n_particles, len(initial_guess))
        )
        
        # Initialize velocities
        velocity_bounds = [(-(b[1] - b[0]), (b[1] - b[0])) for b in bounds]
        velocities = np.random.uniform(
            low=[b[0] for b in velocity_bounds],
            high=[b[1] for b in velocity_bounds],
            size=(n_particles, len(initial_guess))
        )
        
        # Initialize personal best positions and scores
        personal_best_pos = particles.copy()
        personal_best_scores = np.array([
            objective_with_penalty(p, G0, ISP, EPSILON, TOTAL_DELTA_V)
            for p in particles
        ])
        
        # Initialize global best
        global_best_idx = np.argmin(personal_best_scores)
        global_best_pos = personal_best_pos[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]
        
        # Optimization loop
        history = []
        for iteration in range(n_iterations):
            # Update velocities
            r1, r2 = np.random.rand(2, n_particles, len(initial_guess))
            cognitive_velocity = c1 * r1 * (personal_best_pos - particles)
            social_velocity = c2 * r2 * (global_best_pos - particles)
            velocities = w * velocities + cognitive_velocity + social_velocity
            
            # Clip velocities to bounds
            for i, bounds in enumerate(velocity_bounds):
                velocities[:, i] = np.clip(velocities[:, i], bounds[0], bounds[1])
            
            # Update positions
            particles += velocities
            
            # Clip positions to bounds and repair solutions
            for i, bounds in enumerate(bounds):
                particles[:, i] = np.clip(particles[:, i], bounds[0], bounds[1])
            
            # Repair total delta-V constraint
            for i in range(n_particles):
                current_sum = np.sum(particles[i])
                if abs(current_sum - TOTAL_DELTA_V) > 1e-6:
                    scale_factor = TOTAL_DELTA_V / current_sum
                    particles[i] *= scale_factor
            
            # Evaluate new positions
            scores = np.array([
                objective_with_penalty(p, G0, ISP, EPSILON, TOTAL_DELTA_V)
                for p in particles
            ])
            
            # Update personal bests
            improved = scores < personal_best_scores
            personal_best_pos[improved] = particles[improved]
            personal_best_scores[improved] = scores[improved]
            
            # Update global best
            min_score_idx = np.argmin(scores)
            if scores[min_score_idx] < global_best_score:
                global_best_score = scores[min_score_idx]
                global_best_pos = particles[min_score_idx].copy()
            
            # Store history
            history.append({
                'iteration': iteration,
                'best_score': float(global_best_score),
                'mean_score': float(np.mean(scores)),
                'std_score': float(np.std(scores))
            })
            
            # Early stopping if converged
            if iteration > 10 and abs(history[-1]['best_score'] - history[-10]['best_score']) < 1e-8:
                break
        
        # Calculate final metrics
        execution_time = time.time() - start_time
        optimal_dv = global_best_pos
        stage_ratios = calculate_mass_ratios(optimal_dv, ISP, EPSILON, G0)
        payload_fraction = calculate_payload_fraction(stage_ratios)
        
        return {
            'method': 'PSO',
            'optimal_dv': list(optimal_dv),
            'stage_ratios': list(stage_ratios),
            'payload_fraction': float(payload_fraction),
            'execution_time': execution_time,
            'history': history,
            'error': float(abs(np.sum(optimal_dv) - TOTAL_DELTA_V))
        }
        
    except Exception as e:
        logger.error(f"PSO optimization failed: {str(e)}")
        raise
