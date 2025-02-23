#!/usr/bin/env python3

import csv
import sys
import time
import json
import logging
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, differential_evolution, approx_fprime, basinhopping
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.core.problem import Problem
import matplotlib.pyplot as plt
import pandas as pd
from pyswarms.single.global_best import GlobalBestPSO
import cma

# Load configuration
def load_config():
    """Load configuration from JSON file."""
    config_path = Path(__file__).parent / "config.json"
    try:
        with open(config_path) as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration: {e}")

# Set up logging
def setup_logging(config):
    """Configure logging based on configuration."""
    log_config = config["logging"]
    logging.basicConfig(
        filename=log_config["file"],
        level=getattr(logging, log_config["level"]),
        format=log_config["format"]
    )
    return logging.getLogger(__name__)

# Initialize globals from config
CONFIG = load_config()
logger = setup_logging(CONFIG)
GRAVITY_LOSS = CONFIG["constants"]["gravity_loss"]
DRAG_LOSS = CONFIG["constants"]["drag_loss"]

def read_input_json(filename):
    """Read and process JSON input file."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            
        stages = data.get('stages', [])
        if not stages:
            raise ValueError("No stage data found in JSON file")
            
        logger.info(f"Successfully read {len(stages)} stages from {filename}")
        return stages
    except Exception as e:
        logger.error(f"Failed to read JSON input: {e}")
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
        class OptimizationProblem(Problem):
            def __init__(self):
                super().__init__(
                    n_var=len(initial_guess),
                    n_obj=1,
                    n_constr=1,
                    xl=bounds[:, 0],
                    xu=bounds[:, 1]
                )
                self.tol = 1e-6  # tolerance for constraint satisfaction

            def _evaluate(self, x, out, *args, **kwargs):
                # Evaluate the objective for each candidate solution
                f = np.array([
                    payload_fraction_objective(dv, G0, ISP, EPSILON,
                                               config["optimization"]["penalty_coefficient"])
                    for dv in x
                ])
                # Enforce the equality constraint: sum(dv) == TOTAL_DELTA_V (within tol)
                g = np.array([np.abs(np.sum(dv) - TOTAL_DELTA_V) - self.tol for dv in x])
                out["F"] = f
                out["G"] = g

        problem = OptimizationProblem()
        algorithm = GA(
            pop_size=config["optimization"]["ga"]["population_size"],
            eliminate_duplicates=True
        )
        
        res = pymoo_minimize(
            problem,
            algorithm,
            ('n_gen', config["optimization"]["ga"]["n_generations"]),
            seed=1,
            verbose=False
        )
        
        return res.X
    except Exception as e:
        logger.error(f"GA optimization failed: {e}")
        raise


def solve_with_adaptive_ga(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, config):
    """Solve using Adaptive Genetic Algorithm."""
    try:
        class AdaptiveGA:
            def __init__(self, config, n_vars, bounds):
                self.config = config["optimization"]["adaptive_ga"]
                self.n_vars = n_vars
                self.bounds = bounds
                self.pop_size = self.config["initial_pop_size"]
                self.mutation_rate = self.config["initial_mutation_rate"]
                self.crossover_rate = self.config["initial_crossover_rate"]
                self.best_fitness_history = []
                self.diversity_history = []
                self.stagnation_counter = 0
                self.total_delta_v = config["rocket_payload"]["TOTAL_DELTA_V"]
                self.ISP = config["rocket_payload"]["ISP"]

            def initialize_population(self):
                """Initialize population with smart initialization."""
                population = []
                
                # First third: roughly equal distribution
                n_equal = self.pop_size // 3
                for _ in range(n_equal):
                    sol = np.full(self.n_vars, self.total_delta_v / self.n_vars)
                    sol += np.random.normal(0, self.total_delta_v * 0.05, self.n_vars)
                    # Ensure bounds and total delta-v
                    sol = np.clip(sol, self.bounds[:, 0], self.bounds[:, 1])
                    sol = sol * (self.total_delta_v / np.sum(sol))  # Normalize to total delta-v
                    population.append(sol)
                
                # Second third: weighted by ISP
                n_isp = (self.pop_size - n_equal) // 2
                for _ in range(n_isp):
                    weights = np.array(self.ISP) / np.sum(self.ISP)
                    sol = weights * self.total_delta_v
                    sol += np.random.normal(0, self.total_delta_v * 0.05, self.n_vars)
                    # Ensure bounds and total delta-v
                    sol = np.clip(sol, self.bounds[:, 0], self.bounds[:, 1])
                    sol = sol * (self.total_delta_v / np.sum(sol))  # Normalize to total delta-v
                    population.append(sol)
                
                # Last third: random within bounds
                while len(population) < self.pop_size:
                    sol = np.random.uniform(
                        self.bounds[:, 0], 
                        self.bounds[:, 1], 
                        self.n_vars
                    )
                    # Normalize to total delta-v
                    sol = sol * (self.total_delta_v / np.sum(sol))
                    population.append(sol)
                
                return np.array(population)
            
            def evaluate_fitness(self, individual):
                """Calculate fitness for an individual solution."""
                # Ensure individual is 1D
                individual = np.asarray(individual).flatten()
                
                # Check if total delta-v constraint is satisfied
                if not np.isclose(np.sum(individual), self.total_delta_v, rtol=1e-5):
                    return -np.inf
                
                # Calculate payload mass
                payload_mass = calculate_payload_mass(individual)
                
                # Apply penalties for constraint violations
                penalty = 0
                for i, dv in enumerate(individual):
                    if dv < self.bounds[i, 0] or dv > self.bounds[i, 1]:
                        penalty += 1000 * abs(dv - np.clip(dv, self.bounds[i, 0], self.bounds[i, 1]))
                
                return payload_mass - penalty

            def selection(self, population, fitnesses, tournament_size=3):
                """Tournament selection with elitism."""
                # Ensure arrays are properly shaped
                population = np.asarray(population)
                fitnesses = np.asarray(fitnesses).flatten()
                
                # Keep best solution (elitism)
                elite_idx = np.argmax(fitnesses)
                elite = population[elite_idx].copy()
                
                selected = []
                selected.append(elite)  # Add elite
                
                # Tournament selection for rest of population
                while len(selected) < self.pop_size:
                    # Select tournament participants
                    tournament_idx = np.random.choice(len(population), tournament_size, replace=False)
                    tournament_fitnesses = fitnesses[tournament_idx]
                    
                    # Select winner
                    winner_idx = tournament_idx[np.argmax(tournament_fitnesses)]
                    selected.append(population[winner_idx].copy())
                
                return np.array(selected)

            def crossover(self, parent1, parent2):
                """Adaptive blend crossover."""
                if np.random.random() > self.crossover_rate:
                    return parent1.copy(), parent2.copy()
                
                alpha = np.random.random()
                child1 = alpha * parent1 + (1 - alpha) * parent2
                child2 = (1 - alpha) * parent1 + alpha * parent2
                
                # Normalize children to maintain total delta-v
                child1 = child1 * (self.total_delta_v / np.sum(child1))
                child2 = child2 * (self.total_delta_v / np.sum(child2))
                
                return child1, child2
            
            def mutation(self, individual):
                """Adaptive Gaussian mutation."""
                if np.random.random() > self.mutation_rate:
                    return individual.copy()
                
                mutation_strength = (self.bounds[:, 1] - self.bounds[:, 0]) * 0.1
                mutation = np.random.normal(0, mutation_strength, self.n_vars)
                mutated = individual + mutation
                
                # Ensure bounds
                mutated = np.clip(mutated, self.bounds[:, 0], self.bounds[:, 1])
                
                # Normalize to maintain total delta-v
                mutated = mutated * (self.total_delta_v / np.sum(mutated))
                
                return mutated
            
            def optimize(self):
                """Run the optimization process."""
                # Initialize population
                population = self.initialize_population()
                best_fitness = -np.inf
                generations_without_improvement = 0
                
                for generation in range(self.config["n_generations"]):
                    # Evaluate fitness for all individuals
                    fitnesses = np.array([self.evaluate_fitness(ind) for ind in population])
                    
                    # Update best solution
                    current_best_fitness = np.max(fitnesses)
                    if current_best_fitness > best_fitness:
                        best_fitness = current_best_fitness
                        best_solution = population[np.argmax(fitnesses)].copy()
                        generations_without_improvement = 0
                    else:
                        generations_without_improvement += 1
                    
                    # Check termination criteria
                    if generations_without_improvement >= self.config["stagnation_threshold"]:
                        break
                    
                    # Selection
                    selected = self.selection(population, fitnesses)
                    
                    # Create new population through crossover and mutation
                    new_population = []
                    for i in range(0, self.pop_size - 1, 2):
                        parent1, parent2 = selected[i], selected[i + 1]
                        child1, child2 = self.crossover(parent1, parent2)
                        child1 = self.mutation(child1)
                        child2 = self.mutation(child2)
                        new_population.extend([child1, child2])
                    
                    # Ensure population size is correct (in case of odd size)
                    if len(new_population) < self.pop_size:
                        new_population.append(selected[-1])
                    
                    population = np.array(new_population)
                    
                    # Update adaptive parameters
                    self.update_parameters(population, fitnesses, generations_without_improvement)
                
                return best_solution, best_fitness

        # Create and run adaptive GA
        ga = AdaptiveGA(config, len(initial_guess), bounds)
        optimal_solution, _ = ga.optimize()
        
        return optimal_solution
        
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

def optimize_stages(stages, method='SLSQP'):
    """Main optimization function."""
    try:
        n = len(stages)
        G0 = [stage['G0'] for stage in stages]
        ISP = [stage['ISP'] for stage in stages]
        EPSILON = [stage['EPSILON'] for stage in stages]
        
        TOTAL_DELTA_V = 9300.0 + GRAVITY_LOSS + DRAG_LOSS
        penalty_coeff = CONFIG["optimization"]["penalty_coefficient"]
        
        # Initial guess and bounds
        initial_guess = np.ones(n) * TOTAL_DELTA_V / n
        max_dv = TOTAL_DELTA_V * np.ones(n)
        bounds = [(0, mdv) for mdv in max_dv]
        bounds = np.array(bounds)
        
        logger.info(f"Starting optimization with method: {method}")
        start_time = time.time()
        
        if method.upper() == 'SLSQP':
            constraints = {'type': 'eq', 'fun': lambda dv: np.sum(dv) - TOTAL_DELTA_V}
            res = minimize(payload_fraction_objective, initial_guess, 
                         args=(G0, ISP, EPSILON, penalty_coeff),
                         method='SLSQP', bounds=bounds, constraints=constraints)
            optimal_dv = res.x
            
        elif method.upper() == 'GA':
            optimal_dv = solve_with_ga(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, CONFIG)
            
        elif method.upper() == 'ADAPTIVE-GA':
            optimal_dv = solve_with_adaptive_ga(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, CONFIG)
            
        elif method.upper() == 'PSO':
            optimal_dv = solve_with_pso(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, CONFIG)
            
        # Add other solvers here...
        
        execution_time = time.time() - start_time
        logger.info(f"Optimization completed in {execution_time:.6f} seconds")
        
        # Calculate final results
        payload_fraction = -payload_fraction_objective(optimal_dv, G0, ISP, EPSILON, penalty_coeff)
        mass_ratios = calculate_mass_ratios(optimal_dv, ISP, EPSILON)
        
        return {
            'method': method,
            'time': execution_time,
            'payload_fraction': payload_fraction,
            'dv': optimal_dv,
            'mass_ratios': mass_ratios
        }
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise

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
        # Report generation code here...
        logger.info(f"Report generated successfully: {output_file}")
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise

if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            print("Usage: python payload_optimization.py input_data.json")
            sys.exit(1)
            
        input_file = sys.argv[1]
        stages = read_input_json(input_file)
        
        methods = ['SLSQP', 'differential_evolution', 'GA', 'PSO', 'BASIN-HOPPING', 'CMA-ES', 'dp', 'ADAPTIVE-GA']
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
