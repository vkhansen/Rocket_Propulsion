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

def solve_with_genetic_algorithm(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Genetic Algorithm."""
    try:
        class RepairDeltaV(Repair):
            def __init__(self, total_delta_v):
                super().__init__()
                self.total_delta_v = total_delta_v

            def _do(self, problem, X, **kwargs):
                if len(X.shape) == 1:
                    X = X.reshape(1, -1)
                    
                for i in range(len(X)):
                    x = X[i]
                    current_sum = np.sum(x)
                    if abs(current_sum - self.total_delta_v) > 1e-6:
                        # Scale the solution to match total ΔV
                        x = x * (self.total_delta_v / current_sum)
                        # Ensure bounds are satisfied
                        x = np.clip(x, problem.xl, problem.xu)
                        # Re-normalize if clipping changed the sum
                        current_sum = np.sum(x)
                        if abs(current_sum - self.total_delta_v) > 1e-6:
                            x = x * (self.total_delta_v / current_sum)
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
                    
                    # Add penalty for total ΔV constraint
                    penalty = 1e6 * abs(np.sum(dv) - self.total_delta_v)
                    f.append(payload + penalty)
                    
                out["F"] = np.column_stack([f])

        problem = OptimizationProblem(
            n_var=len(initial_guess),
            n_obj=1,
            xl=np.array([b[0] for b in bounds]),
            xu=np.array([b[1] for b in bounds])
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
            repair=RepairDeltaV(TOTAL_DELTA_V)
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
        raise

def solve_with_adaptive_ga(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Adaptive Genetic Algorithm."""
    try:
        logger.info(f"Starting Adaptive GA optimization with parameters:")
        logger.info(f"Initial guess: {initial_guess}")
        logger.info(f"G0: {G0}, ISP: {ISP}, EPSILON: {EPSILON}")
        logger.info(f"TOTAL_DELTA_V: {TOTAL_DELTA_V}")
        
        n_vars = len(initial_guess)
        bounds_array = np.array(bounds)
        
        # Initialize and run the adaptive GA
        ga = AdaptiveGA(config, n_vars, bounds_array, TOTAL_DELTA_V, ISP, EPSILON)
        best_solution = ga.optimize()
        
        if best_solution is not None:
            # Calculate performance metrics
            mass_ratios = calculate_mass_ratios(best_solution, ISP, EPSILON, G0)
            payload_fraction = calculate_payload_fraction(mass_ratios)
            logger.info(f"Adaptive GA optimization succeeded:")
            logger.info(f"  Delta-V: {[f'{dv:.2f}' for dv in best_solution]} m/s")
            logger.info(f"  Mass ratios: {[f'{r:.3f}' for r in mass_ratios]}")
            logger.info(f"  Payload fraction: {payload_fraction:.3f}")
            logger.info(f"  Execution time: {ga.execution_time:.2f} seconds")
            
            # Log optimization history
            logger.info("Optimization history:")
            for entry in ga.history[-5:]:  # Show last 5 generations
                logger.info(f"  Generation {entry['generation']}: "
                          f"Best fitness = {entry['best_fitness']:.3f}, "
                          f"Mean fitness = {entry['mean_fitness']:.3f}, "
                          f"Diversity = {entry['diversity']:.3f}")
            
            return best_solution
        else:
            logger.warning("Adaptive GA optimization failed to find a valid solution")
            return initial_guess
            
    except Exception as e:
        logger.error(f"Adaptive GA optimization failed: {e}")
        raise

class AdaptiveGA:
    def __init__(self, config, n_vars, bounds, total_delta_v, isp, epsilon):
        self.config = config["optimization"]["adaptive_ga"]
        self.n_vars = n_vars
        self.bounds = bounds
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
        population = []
        n_equal = self.pop_size // 3
        for _ in range(n_equal):
            sol = np.full(self.n_vars, self.total_delta_v / self.n_vars)
            sol += np.random.normal(0, self.total_delta_v * 0.05, self.n_vars)
            sol = np.clip(sol, self.bounds[:, 0], self.bounds[:, 1])
            sol = sol * (self.total_delta_v / np.sum(sol))
            population.append(sol)
        n_isp = (self.pop_size - n_equal) // 2
        for _ in range(n_isp):
            weights = np.array(self.ISP) / np.sum(self.ISP)
            sol = weights * self.total_delta_v
            sol += np.random.normal(0, self.total_delta_v * 0.05, self.n_vars)
            sol = np.clip(sol, self.bounds[:, 0], self.bounds[:, 1])
            sol = sol * (self.total_delta_v / np.sum(sol))
            population.append(sol)
        while len(population) < self.pop_size:
            sol = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.n_vars)
            sol = sol * (self.total_delta_v / np.sum(sol))
            population.append(sol)
        return np.array(population)

    def evaluate_fitness(self, individual):
        individual = np.asarray(individual).flatten()
        # Check total delta-v constraint
        if not np.isclose(np.sum(individual), self.total_delta_v, rtol=1e-5):
            return -np.inf
        # Use the payload fraction as the fitness (higher is better)
        mass_ratios = calculate_mass_ratios(individual, self.ISP, self.EPSILON)
        payload_fraction = calculate_payload_fraction(mass_ratios)
        
        # Add penalty for constraint violations
        penalty = 0.0
        for i, dv in enumerate(individual):
            if dv < self.bounds[i, 0] or dv > self.bounds[i, 1]:
                penalty += 1000 * abs(dv - np.clip(dv, self.bounds[i, 0], self.bounds[i, 1]))
        return payload_fraction - penalty

    def selection(self, population, fitnesses, tournament_size=3):
        population = np.asarray(population)
        fitnesses = np.asarray(fitnesses).flatten()
        elite_idx = np.argmax(fitnesses)
        elite = population[elite_idx].copy()
        selected = [elite]
        while len(selected) < self.pop_size:
            tournament_idx = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitnesses = fitnesses[tournament_idx]
            winner_idx = tournament_idx[np.argmax(tournament_fitnesses)]
            selected.append(population[winner_idx].copy())
        return np.array(selected)

    def crossover(self, parent1, parent2):
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        alpha = np.random.random()
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        child1 = child1 * (self.total_delta_v / np.sum(child1))
        child2 = child2 * (self.total_delta_v / np.sum(child2))
        return child1, child2

    def mutation(self, individual):
        if np.random.random() > self.mutation_rate:
            return individual.copy()
        mutation_strength = (self.bounds[:, 1] - self.bounds[:, 0]) * 0.1
        mutation = np.random.normal(0, mutation_strength, self.n_vars)
        mutated = individual + mutation
        mutated = np.clip(mutated, self.bounds[:, 0], self.bounds[:, 1])
        mutated = mutated * (self.total_delta_v / np.sum(mutated))
        return mutated

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

    def calculate_diversity(self, population):
        """Calculate population diversity using standard deviation."""
        return np.mean(np.std(population, axis=0))

    def optimize(self):
        """Run the optimization process."""
        start_time = time.time()
        population = self.initialize_population()
        best_fitness = -np.inf
        best_solution = None
        generations_without_improvement = 0

        for generation in range(self.config["n_generations"]):
            fitnesses = np.array([self.evaluate_fitness(ind) for ind in population])
            current_best_fitness = np.max(fitnesses)
            
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_solution = population[np.argmax(fitnesses)].copy()
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            
            if generations_without_improvement >= self.config["stagnation_threshold"]:
                break
            
            selected = self.selection(population, fitnesses)
            new_population = []
            for i in range(0, self.pop_size - 1, 2):
                parent1, parent2 = selected[i], selected[i+1]
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                new_population.extend([child1, child2])
            
            if len(new_population) < self.pop_size:
                new_population.append(selected[-1])
            
            population = np.array(new_population)
            self.update_parameters(population, fitnesses, generations_without_improvement)
        
        self.execution_time = time.time() - start_time
        return best_solution
