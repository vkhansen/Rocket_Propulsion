#!/usr/bin/env python3

import csv
import sys
import time
import json
import logging
import os
import numpy as np
from scipy.optimize import minimize, basinhopping
import matplotlib.pyplot as plt
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.core.repair import Repair
from pymoo.termination.default import DefaultSingleObjectiveTermination

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

# Ensure output directory exists
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_input_data(filename):
    """Load input data from JSON file."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            
        # Extract global parameters
        global TOTAL_DELTA_V
        TOTAL_DELTA_V = float(data['parameters']['TOTAL_DELTA_V'])
        
        # Sort stages by stage number
        stages = sorted(data['stages'], key=lambda x: x['stage'])
        
        return stages
        
    except Exception as e:
        logger.error(f"Error loading input data: {e}")
        raise

def read_input_json(filename):
    """Read and process JSON input file."""
    try:
        stages = load_input_data(filename)
        
        # Extract global parameters
        global TOTAL_DELTA_V
        TOTAL_DELTA_V = float(stages[0]["parameters"]["TOTAL_DELTA_V"])
        return stages
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        raise

def calculate_mass_ratios(dv, ISP, EPSILON, G0=9.81):
    """Calculate stage ratios for each stage using the correct negative exponent."""
    try:
        dv = np.asarray(dv).flatten()
        mass_ratios = []
        for i, dvi in enumerate(dv):
            # Use a negative exponent as in Code A
            ratio = np.exp(-dvi / (G0 * ISP[i])) - EPSILON[i]
            mass_ratios.append(float(ratio))
        return np.array(mass_ratios)
    except Exception as e:
        logger.error(f"Error calculating mass ratios: {e}")
        return np.array([float('inf')] * len(dv))

def calculate_payload_fraction(mass_ratios):
    """Calculate payload fraction as the product of stage ratios."""
    try:
        if any(r <= 0 for r in mass_ratios):
            return 0.0
        return float(np.prod(mass_ratios))
    except Exception as e:
        logger.error(f"Error calculating payload fraction: {e}")
        return 0.0

def payload_fraction_objective(dv, G0, ISP, EPSILON):
    """Calculate the payload fraction objective using the corrected physics model."""
    try:
        # Pass G0 to calculate_mass_ratios so that the negative exponent is used
        mass_ratios = calculate_mass_ratios(dv, ISP, EPSILON, G0)
        payload_fraction = calculate_payload_fraction(mass_ratios)
        
        # Add a small penalty for solutions close to constraint violations
        penalty = 0.0
        for ratio in mass_ratios:
            if ratio <= 0.1:  # Penalize solutions close to physical limits
                penalty += 100.0 * (0.1 - ratio) ** 2
                
        # Negative for minimization
        return float(-payload_fraction + penalty)
    except Exception as e:
        logger.error(f"Error in payload fraction calculation: {e}")
        return 1e6  # Large but finite penalty

def objective_with_penalty(dv, G0, ISP, EPSILON, TOTAL_DELTA_V):
    """Calculate objective with penalty for constraint violation."""
    try:
        # Base objective
        base_obj = payload_fraction_objective(dv, G0, ISP, EPSILON)
        
        # Constraint violation penalty
        dv_sum = float(np.sum(dv))
        constraint_violation = abs(dv_sum - TOTAL_DELTA_V)
        penalty = 1e3 * constraint_violation  # Reduced penalty coefficient
        
        return float(base_obj + penalty)
    except Exception as e:
        logger.error(f"Error in objective calculation: {e}")
        return 1e6  # Large but finite penalty

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
        payload_fraction = calculate_payload_fraction(mass_ratios)
        
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

def optimize_payload_allocation(TOTAL_DELTA_V, ISP, EPSILON, G0=9.81, method='SLSQP', penalty_coeff=1e6, tol=None):
    """
    Optimize the allocation of TOTAL_DELTA_V among the rocket's stages.
    
    Args:
        TOTAL_DELTA_V (float): Total required delta-V for all stages
        ISP (list): List of specific impulse values for each stage
        EPSILON (list): List of mass fraction values for each stage
        G0 (float): Gravitational constant (default: 9.81 m/s^2)
        method (str): Optimization method to use (default: 'SLSQP')
        penalty_coeff (float): Penalty coefficient for constraint violations (default: 1e6)
        tol (float, optional): Tolerance for constraint satisfaction
    
    Returns:
        tuple: (optimal_dv, stage_ratios, payload_fraction)
    """
    n = len(ISP)
    if n != len(EPSILON):
        raise ValueError("ISP and EPSILON lists must have the same length")
    
    # Initial guess - equal distribution of delta-V
    initial_guess = np.full(n, TOTAL_DELTA_V / n)
    
    # Define bounds for each stage
    max_dv = TOTAL_DELTA_V * 0.9  # No stage should use more than 90% of total dV
    bounds = [(0, max_dv) for _ in range(n)]
    
    # Define the constraint that sum of dV equals TOTAL_DELTA_V
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - TOTAL_DELTA_V}]
    
    try:
        if method.upper() == 'SLSQP':
            result = minimize(
                lambda x: -payload_fraction_objective(x, G0, ISP, EPSILON),
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            optimal_dv = result.x.flatten()
            
        elif method.upper() == 'GA':
            problem = OptimizationProblem(n_var=n, n_obj=1, xl=0, xu=max_dv)
            problem.total_delta_v = TOTAL_DELTA_V
            problem.G0 = G0
            problem.ISP = ISP
            problem.EPSILON = EPSILON
            
            algorithm = GA(pop_size=100)
            res = pymoo_minimize(problem, algorithm, termination=('n_gen', 100))
            optimal_dv = np.array(res.X).flatten()
            
        elif method.upper() == 'BASIN-HOPPING':
            minimizer_kwargs = {
                'method': 'SLSQP',
                'bounds': bounds,
                'constraints': constraints
            }
            res = basinhopping(
                lambda x: -payload_fraction_objective(x, G0, ISP, EPSILON),
                initial_guess,
                minimizer_kwargs=minimizer_kwargs,
                niter=100
            )
            optimal_dv = res.x.flatten()
            
        else:
            raise ValueError(f"Unsupported optimization method: {method}")
        
        # Calculate stage ratios and payload fraction
        stage_ratios = []
        for i, dv in enumerate(optimal_dv):
            ratio = float(np.exp(-dv / (G0 * ISP[i])) - EPSILON[i])
            stage_ratios.append(ratio)
        
        payload_fraction = float(np.prod(stage_ratios))
        optimal_dv = [float(x) for x in optimal_dv]
        
        return optimal_dv, stage_ratios, payload_fraction
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise

def optimize_stages(stages, method='SLSQP'):
    """Run optimization with specified method."""
    try:
        logger.info(f"Starting optimization with method: {method}")
        start_time = time.time()
        
        # Extract stage data
        ISP = [float(stage['ISP']) for stage in stages]
        EPSILON = [float(stage['EPSILON']) for stage in stages]
        G0 = float(stages[0]['G0'])  # Use G0 from first stage
        
        optimal_dv, stage_ratios, payload_fraction = optimize_payload_allocation(
            TOTAL_DELTA_V, ISP, EPSILON, G0, method=method
        )
        
        execution_time = time.time() - start_time
        logger.info(f"Optimization completed in {execution_time:.3f} seconds")
        
        result = {
            'method': method,
            'time': execution_time,
            'payload_fraction': payload_fraction,
            'dv': optimal_dv,  # Already converted to list in optimize_payload_allocation
            'stage_ratios': stage_ratios,  # Already converted to list in optimize_payload_allocation
            'error': 0.0
        }
        
        logger.info(f"Method: {method}, Payload Fraction: {payload_fraction:.4f}, Error: {result['error']:.6e}")
        return result
        
    except Exception as e:
        logger.error(f"Error in {method} optimization: {e}")
        raise

def plot_dv_breakdown(results, filename="dv_breakdown.png"):
    """Plot ΔV breakdown for each optimization method."""
    try:
        plt.figure(figsize=(12, 6))
        
        # Get number of methods
        n_methods = len(results)
        method_positions = np.arange(n_methods)
        bar_width = 0.35
        
        # Create stacked bars for each method
        bottom = np.zeros(n_methods)
        colors = ['dodgerblue', 'orange', 'green']  # Colors for up to 3 stages
        
        # Plot each stage
        n_stages = len(results[0]['dv'])
        for stage in range(n_stages):
            # Extract ΔV values and ratios for this stage across all methods
            stage_dvs = np.array([float(result['dv'][stage]) for result in results])
            stage_ratios = np.array([float(result['stage_ratios'][stage]) for result in results])
            
            # Plot bars for this stage
            plt.bar(method_positions, stage_dvs, bar_width,
                   bottom=bottom, color=colors[stage % len(colors)],
                   label=f'Stage {stage+1}')
            
            # Add text labels with ΔV and λ values
            for i, (dv, ratio) in enumerate(zip(stage_dvs, stage_ratios)):
                plt.text(i, float(bottom[i]) + float(dv)/2,
                        f"{float(dv):.0f} m/s\n(λ={float(ratio):.2f})",
                        ha='center', va='center',
                        color='white', fontweight='bold',
                        fontsize=9)
            
            # Update bottom for next stack
            bottom = bottom + stage_dvs
        
        # Add total text above each bar
        for i, total in enumerate(bottom):
            plt.text(i, float(total) + 100,
                    f"Total: {float(total):.0f} m/s",
                    ha='center', va='bottom',
                    fontweight='bold')
        
        # Add horizontal line for total mission ΔV
        total_dv = float(TOTAL_DELTA_V)
        plt.axhline(y=total_dv, color='red', linestyle='--',
                   label=f'Required ΔV = {total_dv} m/s')
        
        plt.ylabel('ΔV (m/s)')
        plt.xlabel('Optimization Method')
        plt.title('ΔV Solution per Solver')
        plt.xticks(method_positions, [result['method'] for result in results])
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save plot to output directory
        output_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info(f"Plot saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error in plotting ΔV breakdown: {e}")
        raise

def plot_execution_time(results, filename="execution_time.png"):
    """Plot execution time for each optimization method."""
    try:
        plt.figure(figsize=(10, 5))
        method_names = [result['method'] for result in results]
        times = [result['time'] for result in results]
        
        plt.bar(method_names, times)
        plt.title('Solver Execution Time')
        plt.xlabel('Solver Method')
        plt.ylabel('Time (s)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Execution time plot saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error in plotting execution time: {e}")
        raise

def plot_payload_fraction(results, filename="payload_fraction.png"):
    """Plot payload fraction for each optimization method."""
    try:
        plt.figure(figsize=(10, 5))
        method_names = [result['method'] for result in results]
        payloads = [result['payload_fraction'] for result in results]
        
        plt.bar(method_names, payloads)
        plt.title('Payload Fraction by Solver')
        plt.xlabel('Solver Method')
        plt.ylabel('Payload Fraction')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Payload fraction plot saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error in plotting payload fraction: {e}")
        raise

def plot_results(results):
    """Generate all plots."""
    plot_dv_breakdown(results)
    plot_execution_time(results)
    plot_payload_fraction(results)

if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            print(f"Usage: {sys.argv[0]} input_data.json")
            sys.exit(1)
        
        # Load input data
        input_file = sys.argv[1]
        stages = load_input_data(input_file)
        
        # Run optimizations with all methods
        methods = ['SLSQP', 'BASIN-HOPPING', 'GA', 'ADAPTIVE-GA']
        results = []
        
        # Initial setup for optimization
        n = len(stages)
        initial_guess = np.full(n, TOTAL_DELTA_V / n)
        max_dv = TOTAL_DELTA_V * 0.9
        bounds = np.array([(0, max_dv) for _ in range(n)])
        ISP = [float(stage['ISP']) for stage in stages]
        EPSILON = [float(stage['EPSILON']) for stage in stages]
        G0 = float(stages[0]['G0'])

        for method in methods:
            try:
                start_time = time.time()
                
                if method == 'GA':
                    optimal_solution = solve_with_ga(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, CONFIG)
                elif method == 'ADAPTIVE-GA':
                    optimal_solution = solve_with_adaptive_ga(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, CONFIG)
                else:
                    result = optimize_stages(stages, method)
                    results.append(result)
                    continue
                
                # For GA methods, format results consistently with other methods
                execution_time = time.time() - start_time
                mass_ratios = calculate_mass_ratios(optimal_solution, ISP, EPSILON, G0)
                payload_fraction = calculate_payload_fraction(mass_ratios)
                
                result = {
                    'method': method,
                    'time': execution_time,
                    'dv': [float(x) for x in optimal_solution],
                    'stage_ratios': [float(x) for x in mass_ratios],
                    'payload_fraction': float(payload_fraction),
                    'error': 0.0
                }
                results.append(result)
                logger.info(f"Successfully completed {method} optimization")
            except Exception as e:
                logger.error(f"Optimization with {method} failed: {e}")
                continue
        
        # Generate plots
        plot_results(results)
        
    except Exception as e:
        logger.error(f"Program failed: {e}")
        sys.exit(1)
