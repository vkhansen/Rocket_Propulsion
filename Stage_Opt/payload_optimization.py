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

def calculate_mass_ratios(dv, ISP, EPSILON):
    """Calculate mass ratios for each stage."""
    try:
        dv = np.asarray(dv).flatten()
        mass_ratios = []
        for i, dvi in enumerate(dv):
            ratio = np.exp(dvi / (9.81 * ISP[i])) - EPSILON[i]
            mass_ratios.append(float(ratio))
        return np.array(mass_ratios)
    except Exception as e:
        logger.error(f"Error calculating mass ratios: {e}")
        return np.array([float('inf')] * len(dv))

def calculate_payload_fraction(mass_ratios):
    """Calculate payload fraction from mass ratios."""
    try:
        if any(r <= 0 for r in mass_ratios):
            return 0.0
        return float(np.prod(1.0 / mass_ratios))
    except Exception as e:
        logger.error(f"Error calculating payload fraction: {e}")
        return 0.0

def payload_fraction_objective(dv, G0, ISP, EPSILON):
    """Calculate the payload fraction objective."""
    try:
        mass_ratios = calculate_mass_ratios(dv, ISP, EPSILON)
        payload_fraction = calculate_payload_fraction(mass_ratios)
        
        # Add a small penalty for solutions close to constraint violations
        penalty = 0.0
        for ratio in mass_ratios:
            if ratio <= 0.1:  # Penalize solutions close to physical limits
                penalty += 100.0 * (0.1 - ratio)**2
                
        return float(-payload_fraction + penalty)  # Negative for minimization
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
        class OptimizationProblem(Problem):
            def __init__(self, G0, ISP, EPSILON, TOTAL_DELTA_V, **kwargs):
                super().__init__(n_var=len(initial_guess),
                               n_obj=1,
                               n_constr=1,
                               xl=bounds[:, 0],
                               xu=bounds[:, 1],
                               **kwargs)
                self.G0 = G0
                self.ISP = ISP
                self.EPSILON = EPSILON
                self.TOTAL_DELTA_V = TOTAL_DELTA_V
            
            def _evaluate(self, x, out, *args, **kwargs):
                # Handle both single solutions and population
                if len(x.shape) == 1:
                    x = x.reshape(1, -1)
                
                # Calculate objectives
                f = []
                for xi in x:
                    try:
                        val = payload_fraction_objective(xi, self.G0, self.ISP, self.EPSILON)
                        f.append(float(val))
                    except:
                        f.append(1e6)  # Large but finite penalty
                
                out["F"] = np.array(f).reshape(-1, 1)
                
                # Calculate constraints: sum(dv) - TOTAL_DELTA_V = 0
                g = []
                for xi in x:
                    try:
                        val = float(np.sum(xi) - self.TOTAL_DELTA_V)
                        g.append(val)
                    except:
                        g.append(1e6)
                
                out["G"] = np.array(g).reshape(-1, 1)
        
        problem = OptimizationProblem(G0, ISP, EPSILON, TOTAL_DELTA_V)
        
        algorithm = GA(
            pop_size=config["optimization"]["population_size"],
            eliminate_duplicates=True
        )
        
        res = pymoo_minimize(problem,
                           algorithm,
                           termination=('n_gen', config["optimization"]["max_iterations"]),
                           verbose=False)
        
        return res.X  # Return best solution
        
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
            optimal_dv = solve_with_slsqp(initial_guess, bounds, G0, ISP, EPSILON, required_dv, CONFIG)
            
        elif method.upper() == 'differential_evolution':
            optimal_dv = solve_with_differential_evolution(initial_guess, bounds, G0, ISP, EPSILON, required_dv, CONFIG)
            
        elif method.upper() == 'GA':
            optimal_dv = solve_with_ga(initial_guess, bounds, G0, ISP, EPSILON, required_dv, CONFIG)
            
        elif method.upper() == 'PSO':
            optimal_dv = solve_with_pso(initial_guess, bounds, G0, ISP, EPSILON, required_dv, CONFIG)
            
        elif method.upper() == 'BASIN-HOPPING':
            optimal_dv = solve_with_basin_hopping(initial_guess, bounds, G0, ISP, EPSILON, required_dv, CONFIG)
            
        elif method.upper() == 'DP':
            optimal_dv = solve_with_dynamic_programming(initial_guess, bounds, G0, ISP, EPSILON, required_dv, CONFIG)
            
        elif method.upper() == 'ADAPTIVE-GA':
            optimal_dv = solve_with_adaptive_ga(initial_guess, bounds, G0, ISP, EPSILON, required_dv, CONFIG)
            
        else:
            raise NotImplementedError(f"Optimization method {method} is not implemented")

        execution_time = time.time() - start_time
        payload_fraction = -payload_fraction_objective(optimal_dv, G0, ISP, EPSILON)
        mass_ratios = calculate_mass_ratios(optimal_dv, ISP, EPSILON)
        error = abs(np.sum(optimal_dv) - required_dv)
        
        results = {
            'method': method,
            'time': execution_time,
            'payload_fraction': payload_fraction,
            'dv': optimal_dv,
            'mass_ratios': mass_ratios,
            'solution': optimal_dv,
            'error': error
        }
        
        logger.info(f"Optimization completed in {execution_time:.6f} seconds")
        logger.info(f"Successfully optimized using {method}")
        
        return results

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        logger.error(f"Failed to optimize using {method}: {e}")
        raise

def calculate_payload_mass(dv, ISP, EPSILON):
    """Calculate payload mass as the product of stage mass ratios."""
    mass_ratios = calculate_mass_ratios(dv, ISP, EPSILON)
    return np.prod(mass_ratios)

def plot_results(results):
    """Generate plots for optimization results."""
    plot_dv_breakdown(results)
    plot_execution_time(results)
    plot_objective_error(results)

def plot_dv_breakdown(results):
    """Plot ΔV breakdown for each optimization method."""
    try:
        plt.figure(figsize=(12, 6))
        
        # Get number of stages from the first result's dv array
        n_stages = len(results[0]['dv'])
        
        # Set up the bar positions
        n_methods = len(results)
        bar_width = 0.8 / n_stages  # Adjust width based on number of stages
        method_positions = np.arange(n_methods)
        
        # Create bars for each stage
        for i in range(n_stages):
            # Extract ΔV values for this stage across all methods
            stage_dvs = [row['dv'][i] for row in results]
            
            # Calculate position for this stage's bars
            positions = method_positions + (i - n_stages/2 + 0.5) * bar_width
            
            # Plot bars for this stage
            plt.bar(positions, stage_dvs, bar_width, 
                   label=f'Stage {i+1}',
                   alpha=0.7)
        
        plt.xlabel('Optimization Method')
        plt.ylabel('ΔV (m/s)')
        plt.title('ΔV Breakdown by Stage and Method')
        plt.xticks(method_positions, [row['method'] for row in results], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plt.savefig('dv_breakdown.png')
        plt.close()
        
    except Exception as e:
        logger.error(f"Error in plotting ΔV breakdown: {e}")
        raise

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
