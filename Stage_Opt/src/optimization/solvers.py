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
        logger.info(f"Starting Genetic Algorithm optimization with parameters:")
        logger.info(f"Initial guess: {initial_guess}")
        logger.info(f"G0: {G0}, ISP: {ISP}, EPSILON: {EPSILON}")
        logger.info(f"TOTAL_DELTA_V: {TOTAL_DELTA_V}")
        
        # Problem setup with bounds validation
        xl = np.array([max(0.1, b[0]) for b in bounds])  # Ensure positive lower bounds
        xu = np.array([b[1] for b in bounds])
        
        problem = RocketOptimizationProblem(
            n_var=len(initial_guess),
            bounds=bounds,
            G0=G0,
            ISP=ISP,
            EPSILON=EPSILON,
            TOTAL_DELTA_V=TOTAL_DELTA_V
        )
        
        # GA setup with safe parameter values
        algorithm = GA(
            pop_size=config["optimization"]["genetic_algorithm"]["population_size"],
            eliminate_duplicates=True,
            crossover=SBX(
                prob=0.9,  # High crossover probability
                eta=15,    # Distribution index - moderate spread
                repair=DeltaVRepair(TOTAL_DELTA_V),
                vtype=float
            ),
            mutation=PolynomialMutation(
                prob=1.0/len(initial_guess),  # Mutation probability per variable
                eta=20,    # Distribution index - moderate spread
                repair=DeltaVRepair(TOTAL_DELTA_V),
                vtype=float
            )
        )
        
        # Set termination criteria
        termination = DefaultSingleObjectiveTermination(
            xtol=1e-4,
            cvtol=1e-6,
            ftol=1e-4,
            period=20,
            n_max_gen=config["optimization"]["genetic_algorithm"]["max_generations"]
        )
        
        # Run optimization
        start_time = time.time()
        result = pymoo_minimize(
            problem,
            algorithm,
            termination,
            seed=42,
            verbose=False,
            save_history=True
        )
        execution_time = time.time() - start_time
        
        if result.success:
            try:
                # Ensure solution is within bounds and satisfies constraints
                solution = np.clip(result.X, xl, xu)
                total_dv = np.sum(solution)
                if abs(total_dv - TOTAL_DELTA_V) > 1e-6:
                    scale = TOTAL_DELTA_V / total_dv
                    solution = solution * scale
                
                # Calculate performance metrics with error handling
                stage_ratios = calculate_mass_ratios(solution, ISP, EPSILON, G0)
                payload_fraction = calculate_payload_fraction(stage_ratios)
                
                logger.info(f"Genetic Algorithm optimization succeeded:")
                logger.info(f"  Delta-V: {[f'{dv:.2f}' for dv in solution]} m/s")
                logger.info(f"  Mass ratios: {[f'{r:.3f}' for r in stage_ratios]}")
                logger.info(f"  Payload fraction: {payload_fraction:.3f}")
                
                return {
                    'method': 'GA',
                    'optimal_dv': solution.tolist(),
                    'stage_ratios': stage_ratios.tolist(),
                    'payload_fraction': float(payload_fraction),
                    'execution_time': execution_time,
                    'history': [gen.opt[0].F[0] for gen in result.history]
                }
            except Exception as e:
                logger.error(f"Error processing GA results: {e}")
                return None
        else:
            logger.warning(f"Genetic Algorithm optimization failed to converge")
            return None
            
    except Exception as e:
        logger.error(f"Genetic Algorithm optimization failed: {e}")
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
            
            # Check total delta-v constraint
            if not np.isclose(np.sum(individual), self.total_delta_v, rtol=1e-5):
                return -np.inf
                
            # Check bounds constraints
            if np.any(individual < self.bounds_low) or np.any(individual > self.bounds_high):
                return -np.inf
                
            # Calculate payload fraction
            mass_ratios = calculate_mass_ratios(individual, self.ISP, self.EPSILON)
            if mass_ratios is None:
                return -np.inf
            payload_fraction = calculate_payload_fraction(mass_ratios)
            
            return float(payload_fraction)  # Ensure we return a float
            
        except Exception as e:
            logger.error(f"Error in fitness evaluation: {e}")
            return -np.inf

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
                
            # Arithmetic crossover
            alpha = np.random.random()
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = (1 - alpha) * parent1 + alpha * parent2
            
            # Repair to maintain total delta-v
            child1 = np.clip(child1, self.bounds_low, self.bounds_high)
            child2 = np.clip(child2, self.bounds_low, self.bounds_high)
            child1 = child1 * (self.total_delta_v / np.sum(child1))
            child2 = child2 * (self.total_delta_v / np.sum(child2))
            
            return child1, child2
            
        except Exception as e:
            logger.error(f"Error in crossover: {e}")
            return parent1.copy(), parent2.copy()

    def mutation(self, individual):
        """Perform adaptive mutation with repair."""
        try:
            if np.random.random() > self.mutation_rate:
                return individual.copy()
                
            # Calculate adaptive mutation strength
            mutation_strength = (self.bounds_high - self.bounds_low) * 0.1
            mutation = np.random.normal(0, mutation_strength, self.n_vars)
            
            # Apply mutation and repair
            mutated = individual + mutation
            mutated = np.clip(mutated, self.bounds_low, self.bounds_high)
            mutated = mutated * (self.total_delta_v / np.sum(mutated))
            
            return mutated
            
        except Exception as e:
            logger.error(f"Error in mutation: {e}")
            return individual.copy()

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
        try:
            start_time = time.time()
            population = self.initialize_population()
            best_fitness = -np.inf
            best_solution = None
            generations_without_improvement = 0

            for generation in range(self.config["n_generations"]):
                # Evaluate population
                fitnesses = np.array([self.evaluate_fitness(ind) for ind in population])
                current_best_fitness = np.max(fitnesses)
                current_best_idx = np.argmax(fitnesses)
                
                # Update best solution
                if current_best_fitness > best_fitness:
                    best_fitness = current_best_fitness
                    best_solution = population[current_best_idx].copy()
                    generations_without_improvement = 0
                else:
                    generations_without_improvement += 1
                
                # Check for convergence
                if generations_without_improvement >= self.config["stagnation_threshold"]:
                    break
                
                # Evolution
                selected = self.selection(population, fitnesses)
                new_population = []
                
                # Crossover and mutation
                for i in range(0, self.pop_size - 1, 2):
                    parent1, parent2 = selected[i], selected[i+1]
                    child1, child2 = self.crossover(parent1, parent2)
                    child1 = self.mutation(child1)
                    child2 = self.mutation(child2)
                    new_population.extend([child1, child2])
                
                # Handle odd population size
                if len(new_population) < self.pop_size:
                    new_population.append(selected[-1])
                
                population = np.array(new_population)
                self.update_parameters(population, fitnesses, generations_without_improvement)
            
            self.execution_time = time.time() - start_time
            
            if best_solution is not None:
                # Calculate final metrics
                mass_ratios = calculate_mass_ratios(best_solution, self.ISP, self.EPSILON)
                payload_fraction = calculate_payload_fraction(mass_ratios)
                
                return {
                    'method': 'GA-Adaptive',
                    'optimal_dv': best_solution.tolist(),
                    'stage_ratios': mass_ratios.tolist(),
                    'payload_fraction': float(payload_fraction),
                    'execution_time': self.execution_time,
                    'history': self.history
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in optimization: {e}")
            return None

def solve_with_pso(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Particle Swarm Optimization (PSO)."""
    try:
        start_time = time.time()
        n_particles = config["optimization"]["pso"]["n_particles"]
        n_iterations = config["optimization"]["pso"]["n_iterations"]
        c1 = config["optimization"]["pso"]["c1"]  # cognitive parameter
        c2 = config["optimization"]["pso"]["c2"]  # social parameter
        w = config["optimization"]["pso"]["w"]    # inertia weight
        
        n_vars = len(initial_guess)
        
        # Initialize particles and velocities
        particles = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(n_particles, n_vars)
        )
        velocities = np.zeros((n_particles, n_vars))
        
        # Initialize personal best positions and values
        pbest_pos = particles.copy()
        pbest_val = np.array([
            objective_with_penalty(p, G0, ISP, EPSILON, TOTAL_DELTA_V)
            for p in particles
        ])
        
        # Initialize global best
        gbest_idx = np.argmin(pbest_val)
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_val = pbest_val[gbest_idx]
        
        # Optimization history
        history = []
        
        # Main PSO loop
        for _ in range(n_iterations):
            # Update velocities
            r1, r2 = np.random.rand(2)
            velocities = (w * velocities +
                        c1 * r1 * (pbest_pos - particles) +
                        c2 * r2 * (gbest_pos - particles))
            
            # Update positions
            particles += velocities
            
            # Enforce bounds
            particles = np.clip(particles,
                              [b[0] for b in bounds],
                              [b[1] for b in bounds])
            
            # Enforce total delta-v constraint
            sums = np.sum(particles, axis=1)
            scale = TOTAL_DELTA_V / sums
            particles = particles * scale[:, None]
            
            # Evaluate particles
            values = np.array([
                objective_with_penalty(p, G0, ISP, EPSILON, TOTAL_DELTA_V)
                for p in particles
            ])
            
            # Update personal bests
            improved = values < pbest_val
            pbest_pos[improved] = particles[improved]
            pbest_val[improved] = values[improved]
            
            # Update global best
            min_idx = np.argmin(values)
            if values[min_idx] < gbest_val:
                gbest_pos = particles[min_idx].copy()
                gbest_val = values[min_idx]
            
            # Store history
            history.append(gbest_val)
        
        # Calculate final results
        execution_time = time.time() - start_time
        try:
            stage_ratios = calculate_mass_ratios(gbest_pos, ISP, EPSILON, G0)
            payload_fraction = calculate_payload_fraction(stage_ratios)
            
            return {
                'method': 'PSO',
                'optimal_dv': gbest_pos.tolist(),
                'stage_ratios': stage_ratios.tolist(),
                'payload_fraction': float(payload_fraction),
                'execution_time': execution_time,
                'history': history
            }
            
        except Exception as e:
            logger.error(f"Error calculating mass ratios: {e}")
            return None
        
    except Exception as e:
        logger.error(f"Error in PSO solver: {e}")
        return None
