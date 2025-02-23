#!/usr/bin/env python3

import csv
import sys
import time
import json
import logging
import numpy as np
from scipy.optimize import minimize, differential_evolution, basinhopping
import pandas as pd
from pyswarms.single.global_best import GlobalBestPSO
import matplotlib.pyplot as plt
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.core.problem import Problem

# Load configuration
def load_config():
    """Load configuration from config.json."""
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load config.json: {e}")
        sys.exit(1)

def setup_logging(config):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, config["logging"]["level"]),
        format=config["logging"]["format"]
    )
    return logging.getLogger(__name__)

# Initialize globals from config
CONFIG = load_config()
logger = setup_logging(CONFIG)

def read_input_json(filename):
    """Read and process JSON input file."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            
        # Extract global parameters
        global TOTAL_DELTA_V
        TOTAL_DELTA_V = float(data["parameters"]["TOTAL_DELTA_V"])
        return data["stages"]
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        raise

def payload_fraction_objective(dv, G0, ISP, EPSILON, penalty_coeff):
    """Calculate payload fraction objective."""
    try:
        n = len(dv)
        mf = np.ones(n)  # mass fractions
        for i in range(n-1, -1, -1):
            mf[i] = np.exp(-dv[i]/(ISP[i]*9.81))
        
        # Calculate stage mass ratios
        mass_ratios = []
        for i in range(n):
            eps = EPSILON[i]
            mf_i = mf[i]
            ratio = eps/(1-mf_i*(1-eps))
            mass_ratios.append(ratio)
            
        payload_fraction = np.prod(mass_ratios)
        return -payload_fraction  # Negative because we want to maximize
    except Exception as e:
        logger.error(f"Error in payload fraction calculation: {e}")
        return float('inf')

def objective_with_penalty(dv, G0, ISP, EPSILON, TOTAL_DELTA_V, penalty_coeff, tol=1e-6):
    """Objective function with penalty for constraint violation."""
    try:
        constraint_violation = abs(np.sum(dv) - TOTAL_DELTA_V)
        obj_value = payload_fraction_objective(dv, G0, ISP, EPSILON, penalty_coeff)
        penalty = penalty_coeff * constraint_violation
        
        if constraint_violation > tol:
            logger.warning(f"Constraint violation: {constraint_violation}")
            
        return obj_value + penalty
    except Exception as e:
        logger.error(f"Error in penalty calculation: {e}")
        return float('inf')

def solve_with_ga(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Genetic Algorithm."""
    try:
        start_time = time.time()
        
        class OptimizationProblem:
            def __init__(self):
                self.n_var = len(initial_guess)
                self.xl = bounds[:, 0]
                self.xu = bounds[:, 1]
                self.tol = 1e-6  # tolerance for constraint satisfaction
                self.history = []  # Store optimization history

            def _evaluate(self, x, out, *args, **kwargs):
                # Evaluate the objective for each candidate solution
                f = np.array([
                    payload_fraction_objective(dv, G0, ISP, EPSILON,
                                               config["optimization"]["penalty_coefficient"])
                    for dv in x
                ])
                # Enforce the equality constraint: sum(dv) == TOTAL_DELTA_V (within tol)
                g = np.array([np.abs(np.sum(dv) - TOTAL_DELTA_V) - self.tol for dv in x])
                
                # Record current state
                best_idx = np.argmin(f)
                self.history.append({
                    'generation': len(self.history),
                    'best_fitness': float(f[best_idx]),
                    'mean_fitness': float(np.mean(f)),
                    'best_solution': x[best_idx].copy(),
                    'constraint_violation': float(g[best_idx])
                })
                
                out["F"] = f
                out["G"] = g

        # Setup and run GA
        problem = OptimizationProblem()
        from pymoo.algorithms.soo.nonconvex.ga import GA
        algorithm = GA(
            pop_size=config["optimization"]["ga"]["population_size"],
            eliminate_duplicates=True
        )
        
        res = minimize(
            problem,
            algorithm,
            ('n_gen', config["optimization"]["ga"]["n_generations"]),
            seed=1,
            verbose=False
        )
        
        execution_time = time.time() - start_time
        optimal_solution = res.X
        
        # Calculate metrics for the optimal solution
        mass_ratios = calculate_mass_ratios(optimal_solution, ISP, EPSILON)
        payload_fraction = calculate_payload_mass(optimal_solution, ISP, EPSILON)
        
        # Return results in format expected by plotting functions
        return {
            'method': 'GA',
            'time': execution_time,
            'solution': optimal_solution,
            'fitness': float(res.F[0]),
            'mass_ratios': mass_ratios,
            'payload_fraction': payload_fraction,
            'dv': optimal_solution,  # For delta-v breakdown plot
            'error': float(res.G[0]),  # Constraint violation
            'history': problem.history  # For convergence plots
        }
        
    except Exception as e:
        logger.error(f"GA optimization failed: {e}")
        raise

def solve_with_adaptive_ga(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Adaptive Genetic Algorithm."""
    try:
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
                # Use the payload mass (fraction) as the fitness (higher is better)
                payload_mass = calculate_payload_mass(individual, self.ISP, self.EPSILON)
                penalty = 0
                for i, dv in enumerate(individual):
                    if dv < self.bounds[i, 0] or dv > self.bounds[i, 1]:
                        penalty += 1000 * abs(dv - np.clip(dv, self.bounds[i, 0], self.bounds[i, 1]))
                return payload_mass - penalty

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
                if diversity < self.config.get("diversity_threshold", 0.1):
                    self.mutation_rate = min(self.mutation_rate * 1.5, 0.5)
                else:
                    self.mutation_rate = max(self.mutation_rate * 0.9, 0.01)
                
                # Adjust crossover rate based on fitness improvement
                if generations_without_improvement > 5:
                    self.crossover_rate = min(self.crossover_rate * 1.1, 0.95)
                else:
                    self.crossover_rate = max(self.crossover_rate * 0.9, 0.5)

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
                return best_solution, best_fitness

        # Create and run adaptive GA, passing TOTAL_DELTA_V, ISP, and EPSILON
        ga = AdaptiveGA(config, len(initial_guess), bounds, TOTAL_DELTA_V, ISP, EPSILON)
        optimal_solution, best_fitness = ga.optimize()
        
        # Calculate metrics for the optimal solution
        mass_ratios = calculate_mass_ratios(optimal_solution, ISP, EPSILON)
        payload_fraction = calculate_payload_mass(optimal_solution, ISP, EPSILON)
        
        # Return results in format expected by plotting functions
        return {
            'method': 'ADAPTIVE-GA',
            'time': ga.execution_time,
            'solution': optimal_solution,
            'fitness': best_fitness,
            'mass_ratios': mass_ratios,
            'payload_fraction': payload_fraction,
            'dv': optimal_solution,  # For delta-v breakdown plot
            'error': abs(np.sum(optimal_solution) - TOTAL_DELTA_V),  # Constraint violation
            'history': ga.history  # For convergence plots
        }
        
    except Exception as e:
        logger.error(f"Adaptive GA optimization failed: {e}")
        raise

def solve_with_pso(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Particle Swarm Optimization."""
    try:
        pso_config = config["optimization"]["pso"]
        def pso_objective(x):
            return np.array([
                objective_with_penalty(pos, G0, ISP, EPSILON, TOTAL_DELTA_V, 
                                    config["optimization"]["penalty_coefficient"])
                for pos in x
            ])

        optimizer = GlobalBestPSO(
            n_particles=pso_config["n_particles"],
            dimensions=len(initial_guess),
            options={
                'c1': pso_config["c1"],
                'c2': pso_config["c2"],
                'w': pso_config["w"]
            },
            bounds=(bounds[:, 0], bounds[:, 1])
        )
        
        best_cost, best_pos = optimizer.optimize(
            pso_objective,
            iters=pso_config["n_iterations"]
        )
        return best_pos
    except Exception as e:
        logger.error(f"PSO optimization failed: {e}")
        raise

def solve_with_differential_evolution(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Differential Evolution."""
    try:
        start_time = time.time()
        penalty_coeff = config["optimization"]["penalty_coefficient"]
        
        def objective(dv):
            return payload_fraction_objective(dv, G0, ISP, EPSILON, penalty_coeff)
        
        result = differential_evolution(objective, bounds, 
                                     popsize=config["optimization"]["population_size"],
                                     maxiter=config["optimization"]["max_iterations"],
                                     tol=config["optimization"]["tolerance"])
        
        optimal_dv = result.x
        execution_time = time.time() - start_time
        payload_fraction = -objective(optimal_dv)
        mass_ratios = calculate_mass_ratios(optimal_dv, ISP, EPSILON)
        
        return {
            'time': execution_time,
            'payload_fraction': payload_fraction,
            'dv': optimal_dv,
            'mass_ratios': mass_ratios,
            'solution': optimal_dv,
            'error': abs(np.sum(optimal_dv) - TOTAL_DELTA_V)
        }
        
    except Exception as e:
        logger.error(f"Differential Evolution optimization failed: {e}")
        raise

def solve_with_basin_hopping(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Basin-Hopping."""
    try:
        start_time = time.time()
        penalty_coeff = config["optimization"]["penalty_coefficient"]
        
        def objective(dv):
            return payload_fraction_objective(dv, G0, ISP, EPSILON, penalty_coeff)
        
        minimizer_kwargs = {
            "method": "L-BFGS-B",
            "bounds": bounds
        }
        
        result = basinhopping(objective, initial_guess,
                            niter=config["optimization"]["max_iterations"],
                            minimizer_kwargs=minimizer_kwargs,
                            stepsize=0.1)
        
        optimal_dv = result.x
        execution_time = time.time() - start_time
        payload_fraction = -objective(optimal_dv)
        mass_ratios = calculate_mass_ratios(optimal_dv, ISP, EPSILON)
        
        return {
            'time': execution_time,
            'payload_fraction': payload_fraction,
            'dv': optimal_dv,
            'mass_ratios': mass_ratios,
            'solution': optimal_dv,
            'error': abs(np.sum(optimal_dv) - TOTAL_DELTA_V)
        }
        
    except Exception as e:
        logger.error(f"Basin-Hopping optimization failed: {e}")
        raise

def solve_with_nelder_mead(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Nelder-Mead with bounds handling."""
    try:
        start_time = time.time()
        penalty_coeff = config["optimization"]["penalty_coefficient"]
        
        def objective(x):
            # Apply bounds through penalty
            lower_bounds = bounds[:, 0]
            upper_bounds = bounds[:, 1]
            
            # Calculate bounds violation penalty
            lower_violation = np.sum(np.maximum(0, lower_bounds - x)**2)
            upper_violation = np.sum(np.maximum(0, x - upper_bounds)**2)
            bounds_penalty = 1e6 * (lower_violation + upper_violation)
            
            # Calculate original objective
            obj_value = payload_fraction_objective(x, G0, ISP, EPSILON, penalty_coeff)
            
            return obj_value + bounds_penalty
        
        # Run Nelder-Mead optimization
        result = minimize(objective, initial_guess, method='Nelder-Mead',
                        options={'maxiter': config["optimization"]["max_iterations"],
                                'xatol': config["optimization"]["tolerance"],
                                'fatol': config["optimization"]["tolerance"]})
        
        # Ensure result is within bounds
        optimal_dv = np.clip(result.x, bounds[:, 0], bounds[:, 1])
        execution_time = time.time() - start_time
        payload_fraction = -payload_fraction_objective(optimal_dv, G0, ISP, EPSILON, penalty_coeff)
        mass_ratios = calculate_mass_ratios(optimal_dv, ISP, EPSILON)
        
        return {
            'time': execution_time,
            'payload_fraction': payload_fraction,
            'dv': optimal_dv,
            'mass_ratios': mass_ratios,
            'solution': optimal_dv,
            'error': abs(np.sum(optimal_dv) - TOTAL_DELTA_V)
        }
        
    except Exception as e:
        logger.error(f"Nelder-Mead optimization failed: {e}")
        raise

def solve_with_dynamic_programming(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Dynamic Programming."""
    try:
        start_time = time.time()
        penalty_coeff = config["optimization"]["penalty_coefficient"]
        
        # DP grid parameters
        n_points = 100  # Number of grid points
        n_stages = len(initial_guess)
        
        # Create grid for each stage
        grid = np.linspace(0, TOTAL_DELTA_V, n_points)
        value_table = np.zeros((n_stages, n_points))
        decision_table = np.zeros((n_stages, n_points))
        
        # Fill tables using backward recursion
        for stage in range(n_stages-1, -1, -1):
            for i, dv in enumerate(grid):
                remaining_dv = TOTAL_DELTA_V - dv
                if stage == n_stages-1:
                    if abs(remaining_dv) <= 1e-6:  # Last stage must use remaining dv
                        value_table[stage, i] = -payload_fraction_objective(
                            np.array([dv]), [G0[stage]], [ISP[stage]], [EPSILON[stage]], penalty_coeff
                        )
                    continue
                
                # Try different allocations of remaining dv
                best_value = -np.inf
                best_next_dv = 0
                for j, next_dv in enumerate(grid):
                    if next_dv > remaining_dv:
                        break
                    
                    current_value = -payload_fraction_objective(
                        np.array([dv]), [G0[stage]], [ISP[stage]], [EPSILON[stage]], penalty_coeff
                    )
                    total_value = current_value + value_table[stage+1, j]
                    
                    if total_value > best_value:
                        best_value = total_value
                        best_next_dv = next_dv
                
                value_table[stage, i] = best_value
                decision_table[stage, i] = best_next_dv
        
        # Reconstruct optimal solution
        optimal_dv = np.zeros(n_stages)
        current_dv = 0
        for stage in range(n_stages):
            idx = np.abs(grid - current_dv).argmin()
            optimal_dv[stage] = decision_table[stage, idx]
            current_dv = optimal_dv[stage]
        
        execution_time = time.time() - start_time
        payload_fraction = -payload_fraction_objective(optimal_dv, G0, ISP, EPSILON, penalty_coeff)
        mass_ratios = calculate_mass_ratios(optimal_dv, ISP, EPSILON)
        
        return {
            'time': execution_time,
            'payload_fraction': payload_fraction,
            'dv': optimal_dv,
            'mass_ratios': mass_ratios,
            'solution': optimal_dv,
            'error': abs(np.sum(optimal_dv) - TOTAL_DELTA_V)
        }
        
    except Exception as e:
        logger.error(f"Dynamic Programming optimization failed: {e}")
        raise

def optimize_stages(stages, method='SLSQP'):
    try:
        n = len(stages)
        G0 = [stage['G0'] for stage in stages]
        ISP = [stage['ISP'] for stage in stages]
        EPSILON = [stage['EPSILON'] for stage in stages]

        required_dv = TOTAL_DELTA_V
        penalty_coeff = CONFIG["optimization"]["penalty_coefficient"]

        initial_guess = np.ones(n) * required_dv / n
        max_dv = required_dv * CONFIG["optimization"]["bounds"]["max_dv_factor"] * np.ones(n)
        min_dv = CONFIG["optimization"]["bounds"]["min_dv"] * np.ones(n)
        bounds = np.array([(min_dv[i], max_dv[i]) for i in range(n)])

        logger.info(f"Starting optimization with method: {method}")
        start_time = time.time()

        if method.upper() == 'SLSQP':
            constraints = {'type': 'eq', 'fun': lambda dv: np.sum(dv) - required_dv}
            res = minimize(payload_fraction_objective, initial_guess,
                           args=(G0, ISP, EPSILON, penalty_coeff),
                           method='SLSQP', bounds=bounds, constraints=constraints)
            optimal_dv = res.x
            execution_time = time.time() - start_time
            payload_fraction = -payload_fraction_objective(optimal_dv, G0, ISP, EPSILON, penalty_coeff)
            mass_ratios = calculate_mass_ratios(optimal_dv, ISP, EPSILON)
            
            results = {
                'method': method,
                'time': execution_time,
                'payload_fraction': payload_fraction,
                'dv': optimal_dv,
                'mass_ratios': mass_ratios,
                'solution': optimal_dv,
                'error': abs(np.sum(optimal_dv) - required_dv)
            }

        elif method.upper() == 'GA':
            results = solve_with_ga(initial_guess, bounds, G0, ISP, EPSILON, required_dv, CONFIG)
            results['method'] = method

        elif method.upper() == 'ADAPTIVE-GA':
            results = solve_with_adaptive_ga(initial_guess, bounds, G0, ISP, EPSILON, required_dv, CONFIG)
            results['method'] = method

        elif method.upper() == 'PSO':
            optimal_dv = solve_with_pso(initial_guess, bounds, G0, ISP, EPSILON, required_dv, CONFIG)
            execution_time = time.time() - start_time
            payload_fraction = -payload_fraction_objective(optimal_dv, G0, ISP, EPSILON, penalty_coeff)
            mass_ratios = calculate_mass_ratios(optimal_dv, ISP, EPSILON)
            
            results = {
                'method': method,
                'time': execution_time,
                'payload_fraction': payload_fraction,
                'dv': optimal_dv,
                'mass_ratios': mass_ratios,
                'solution': optimal_dv,
                'error': abs(np.sum(optimal_dv) - required_dv)
            }

        elif method.upper() == 'DIFFERENTIAL_EVOLUTION':
            results = solve_with_differential_evolution(initial_guess, bounds, G0, ISP, EPSILON, required_dv, CONFIG)
            results['method'] = method

        elif method.upper() == 'BASIN-HOPPING':
            results = solve_with_basin_hopping(initial_guess, bounds, G0, ISP, EPSILON, required_dv, CONFIG)
            results['method'] = method

        elif method.upper() == 'DP':
            results = solve_with_dynamic_programming(initial_guess, bounds, G0, ISP, EPSILON, required_dv, CONFIG)
            results['method'] = method

        else:
            raise NotImplementedError(f"Optimization method {method} is not implemented")

        logger.info(f"Optimization completed in {results['time']:.6f} seconds")
        return results

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise

def calculate_payload_mass(dv, ISP, EPSILON):
    """Calculate payload mass as the product of stage mass ratios."""
    mass_ratios = calculate_mass_ratios(dv, ISP, EPSILON)
    return np.prod(mass_ratios)

def calculate_mass_ratios(dv, ISP, EPSILON):
    """Calculate mass ratios for given solution."""
    try:
        n = len(dv)
        mf = np.ones(n)
        for i in range(n-1, -1, -1):
            mf[i] = np.exp(-dv[i]/(ISP[i]*9.81))
        
        mass_ratios = []
        for i in range(n):
            eps = EPSILON[i]
            mf_i = mf[i]
            ratio = eps/(1-mf_i*(1-eps))
            mass_ratios.append(ratio)
        
        return mass_ratios
    except Exception as e:
        logger.error(f"Mass ratio calculation failed: {e}")
        raise

def generate_report(results, output_file):
    """Generate LaTeX report."""
    try:
        # Generate plots
        plot_results(results)
        
        # Generate LaTeX report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\\documentclass{article}\n')
            f.write('\\usepackage{graphicx}\n')  # For including images
            f.write('\\begin{document}\n')
            f.write('\\section{Optimization Results}\n')
            
            # Add results table
            f.write('\\begin{table}[h]\n')
            f.write('\\centering\n')
            f.write('\\begin{tabular}{|l|c|c|c|}\n')
            f.write('\\hline\n')
            f.write('Method & Time (s) & Payload Fraction & Error \\\\\n')
            f.write('\\hline\n')
            
            for result in results:
                method = result['method'].replace('_', '\\_')  # Escape underscores for LaTeX
                f.write(f"{method} & {result['time']:.3f} & {result['payload_fraction']:.4f} & {result.get('error', 'N/A')} \\\\\n")
            
            f.write('\\hline\n')
            f.write('\\end{tabular}\n')
            f.write('\\caption{Optimization Results}\n')
            f.write('\\end{table}\n')
            
            # Include plots
            f.write('\\begin{figure}[h]\n')
            f.write('\\centering\n')
            f.write('\\includegraphics[width=0.8\\textwidth]{dv_breakdown.png}\n')
            f.write('\\caption{$\\Delta$V Breakdown per Method}\n')
            f.write('\\end{figure}\n')
            
            f.write('\\end{document}\n')
        
        logger.info("Report generated successfully: " + output_file)
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise

def plot_results(results):
    """Generate plots for optimization results."""
    plot_dv_breakdown(results)
    plot_execution_time(results)
    plot_objective_error(results)

def plot_dv_breakdown(results, filename="dv_breakdown.png"):
    """Plot ΔV breakdown for each optimization method."""
    solver_names = [res["method"] for res in results]
    indices = np.arange(len(solver_names))
    bar_width = 0.15

    required_engine_dv = TOTAL_DELTA_V

    plt.figure(figsize=(12, 6))
    for i, res in enumerate(results):
        dv = res["dv"]
        mass_ratios = res["mass_ratios"]
        total_dv = np.sum(dv)
        
        # Plot stacked bars for each stage
        plt.bar(i, dv[0], bar_width, label='Stage 1' if i == 0 else "", color='dodgerblue')
        plt.bar(i, dv[1], bar_width, bottom=dv[0], label='Stage 2' if i == 0 else "", color='orange')
        
        # Add total value on top
        plt.text(i, total_dv + 50, f"{total_dv:.0f}", ha='center', va='bottom', fontsize=9)
    
    plt.axhline(required_engine_dv, color='red', linestyle='--', linewidth=2, label='Required Engine ΔV')
    plt.xticks(indices, solver_names)
    plt.ylabel("Engine-Provided ΔV (m/s)")
    plt.title("ΔV Breakdown per Method")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_execution_time(results, filename="execution_time.png"):
    """Plot execution time for each optimization method."""
    solver_names = [res["method"] for res in results]
    times = [res["time"] for res in results]
    
    plt.figure(figsize=(10, 5))
    plt.bar(solver_names, times, color='skyblue', alpha=0.8)
    plt.xlabel("Optimization Method")
    plt.ylabel("Execution Time (s)")
    plt.title("Solver Execution Time")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_objective_error(results, filename="objective_error.png"):
    """Plot objective error for each optimization method."""
    solver_names = [res["method"] for res in results]
    errors = [res.get("error", np.nan) for res in results]  # Use np.nan for methods without error
    
    plt.figure(figsize=(10, 5))
    plt.bar(solver_names, errors, color='salmon', alpha=0.8)
    plt.xlabel("Optimization Method")
    plt.ylabel("Final Objective Error")
    plt.title("Solver Accuracy (Lower is Better)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            print("Usage: python payload_optimization.py input_data.json")
            sys.exit(1)
            
        input_file = sys.argv[1]
        stages = read_input_json(input_file)
        
        # Run all optimization methods
        methods = ['SLSQP', 'differential_evolution', 'GA', 'PSO', 'BASIN-HOPPING', 'DP', 'ADAPTIVE-GA']
        optimization_results = []
        
        for method in methods:
            try:
                result = optimize_stages(stages, method=method)
                optimization_results.append(result)
                logger.info(f"Successfully optimized using {method}")
            except Exception as e:
                logger.error(f"Failed to optimize using {method}: {e}")
                continue
        
        generate_report(optimization_results, "report.tex")
        
    except Exception as e:
        logger.error(f"Program failed: {e}")
        sys.exit(1)
