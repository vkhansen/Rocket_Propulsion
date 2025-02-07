#!/usr/bin/env python
"""
Payload Optimization with CSV Input

This script performs payload optimization for rockets with any number of stages.
All input data is read from a CSV file formatted as follows:

The CSV file contains two sections separated by a blank line.
1. Global parameters (2 columns: Parameter, Value):
     Parameter,Value
     G0,9.81
     TOTAL_DELTA_V,9500

2. Stage data (3 columns: stage, ISP, EPSILON):
     stage,ISP,EPSILON
     1,300,0.1
     2,320,0.08

Usage:
    python payload_optimization_csv.py input_data.csv
"""

import csv
import sys
import time  # for timing
import numpy as np
from scipy.optimize import minimize, differential_evolution
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.core.problem import Problem
import matplotlib.pyplot as plt

# Example constants for gravity and drag losses (in m/s)
GRAVITY_LOSS = 100.0  
DRAG_LOSS = 50.0      

def read_csv_input(filename):
    """Reads the CSV file and returns a dictionary of global parameters and a list of stage data."""
    parameters = {}
    stages = []
    mode = "parameters"
    
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row or all(cell.strip() == "" for cell in row):
                if mode == "parameters":
                    mode = "stages"
                continue

            if mode == "parameters":
                if row[0].strip().lower() == "parameter":
                    continue  # Skip header
                if len(row) < 2:
                    continue
                parameters[row[0].strip()] = row[1].strip()
            else:  # mode == "stages"
                if row[0].strip().lower() == "stage":
                    continue  # Skip header
                if len(row) < 3:
                    continue
                try:
                    stages.append({
                        'stage': int(row[0].strip()),
                        'ISP': float(row[1].strip()),
                        'EPSILON': float(row[2].strip())
                    })
                except Exception as e:
                    print(f"Error parsing stage row: {row}, {e}")
    return parameters, sorted(stages, key=lambda x: x['stage'])

def payload_fraction_objective(dv, G0, ISP, EPSILON):
    """
    Objective function to be minimized.
    Flattens the input so that dv is a 1D array of scalars.
    Returns the negative product of stage ratios.
    """
    dv = np.asarray(dv).flatten()  # Ensure dv is 1D
    product = 1.0
    for i, dvi in enumerate(dv):
        # Calculate the stage ratio for this stage.
        f_i = np.exp(-dvi / (G0 * ISP[i])) - EPSILON[i]
        # Return a high penalty if the stage ratio is not positive.
        if f_i <= 0:
            return 1e6  
        product *= f_i
    return -product

def objective_with_penalty(dv, G0, ISP, EPSILON, total_delta_v, penalty_coeff=1e6, tol=1e-6):
    """
    A wrapper objective that adds a penalty if the equality constraint
        sum(dv) == total_delta_v
    is violated.
    """
    dv = np.asarray(dv).flatten()
    constraint_error = abs(np.sum(dv) - total_delta_v)
    penalty = penalty_coeff * constraint_error if constraint_error > tol else 0.0
    return payload_fraction_objective(dv, G0, ISP, EPSILON) + penalty

def optimize_payload_allocation(TOTAL_DELTA_V, ISP, EPSILON, G0=9.81, method='SLSQP'):
    """Optimize the allocation of TOTAL_DELTA_V among the rocket's stages."""
    n = len(ISP)
    if n != len(EPSILON):
        raise ValueError("ISP and EPSILON lists must have the same length.")
    
    # Maximum delta-V per stage based on the constraints.
    max_dv = np.array([-G0 * isp * np.log(eps) for isp, eps in zip(ISP, EPSILON)])
    if TOTAL_DELTA_V > np.sum(max_dv):
        raise ValueError("TOTAL_DELTA_V exceeds the maximum possible with the given stage constraints.")
    
    if n == 1:
        dv = TOTAL_DELTA_V
        f = np.exp(-dv / (G0 * ISP[0])) - EPSILON[0]
        return [dv], [f], f
    
    initial_guess = np.full(n, TOTAL_DELTA_V / n)
    bounds = [(0, max_dv_i) for max_dv_i in max_dv]
    
    # For SLSQP we can use the constraint directly.
    if method == 'SLSQP':
        constraints = {'type': 'eq', 'fun': lambda dv: np.sum(dv) - TOTAL_DELTA_V}
        res = minimize(payload_fraction_objective, initial_guess, args=(G0, ISP, EPSILON), 
                       method=method, bounds=bounds, constraints=constraints)
    elif method == 'differential_evolution':
        # Use the penalized objective for DE.
        res = differential_evolution(
            objective_with_penalty, 
            bounds, 
            args=(G0, ISP, EPSILON, TOTAL_DELTA_V),
            strategy='best1bin',
            popsize=200,      # Increased population size may improve search
            tol=1e-6,
            mutation=(0.5, 1),
            recombination=0.7,
            polish=True
        )
    elif method == 'GA':
        # Use the penalized objective within the GA evaluation.
        class OptimizationProblem(Problem):
            def __init__(self):
                super().__init__(n_var=n, n_obj=1, xl=np.zeros(n), xu=max_dv)
            def _evaluate(self, x, out, *args, **kwargs):
                # Ensure x is 2D.
                if x.ndim == 1:
                    x = x.reshape(1, -1)
                F = np.array([
                    objective_with_penalty(np.asarray(xi).flatten(), G0, ISP, EPSILON, TOTAL_DELTA_V)
                    for xi in x
                ])
                out["F"] = F.reshape(-1, 1)
        
        problem = OptimizationProblem()
        # Increase population size and number of generations.
        algorithm = GA(pop_size=100)
        res_obj = pymoo_minimize(problem, algorithm, termination=('n_gen', 100))
        res = res_obj.X  # GA returns the best solution directly

    # For SLSQP and differential_evolution, check that optimization was successful.
    if method in ['SLSQP', 'differential_evolution']:
        if not res.success:
            raise RuntimeError(f"Optimization failed: {res.message}")
        optimal_dv = res.x
    else:
        optimal_dv = res

    optimal_stage_ratio = [np.exp(-dvi / (G0 * ISP[i])) - EPSILON[i] for i, dvi in enumerate(optimal_dv)]
    overall_payload = np.prod(optimal_stage_ratio)
    
    return optimal_dv, optimal_stage_ratio, overall_payload

def plot_dv_breakdown(results, total_delta_v, gravity_loss, drag_loss, filename="dv_breakdown.png"):
    """
    Plot the engine-provided ΔV breakdown per solver, including drag and gravity losses.
    For each solver (result), the delta-V allocated per stage is plotted as stacked bars.
    """
    solver_names = [res["method"] for res in results]
    indices = np.arange(len(solver_names))
    bar_width = 0.15

    required_engine_dv = total_delta_v + gravity_loss + drag_loss

    plt.figure(figsize=(12, 6))
    colors = ['dodgerblue', 'orange', 'green', 'purple', 'cyan', 'magenta']
    for i, res in enumerate(results):
        sol = res["dv"]  # The allocated ΔV per stage.
        cumulative = 0
        for j, dv in enumerate(sol):
            plt.bar(i, dv, bar_width, bottom=cumulative, color=colors[j % len(colors)],
                    label=f'Stage {j+1}' if i == 0 else "")
            cumulative += dv
        plt.text(i, cumulative + 50, f"{cumulative:.0f}", ha='center', va='bottom', fontsize=9)
    plt.axhline(required_engine_dv, color='red', linestyle='--', linewidth=2, label='Required Engine ΔV')
    plt.xticks(indices, solver_names)
    plt.ylabel("Engine-Provided ΔV (m/s)")
    plt.title("ΔV Breakdown per Solver")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} input_data.csv")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    parameters, stages = read_csv_input(input_csv)
    
    try:
        G0 = float(parameters.get("G0", 9.81))
        TOTAL_DELTA_V = float(parameters["TOTAL_DELTA_V"])
    except KeyError as e:
        print(f"Missing required parameter in CSV file: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Invalid parameter value: {e}")
        sys.exit(1)
    
    ISP = [stage['ISP'] for stage in stages]
    EPSILON = [stage['EPSILON'] for stage in stages]

    methods = ['SLSQP', 'differential_evolution', 'GA']
    results = []

    # Loop over solvers and store each result.
    for method in methods:
        start_time = time.time()
        try:
            optimal_dv, optimal_stage_ratio, overall_payload = optimize_payload_allocation(
                TOTAL_DELTA_V, ISP, EPSILON, G0, method=method
            )
            execution_time = time.time() - start_time
            results.append({
                'method': method,
                'time': execution_time,
                'dv': optimal_dv,
                'ratio': optimal_stage_ratio,
                'payload': overall_payload
            })
        except Exception as e:
            print(f"Error during optimization with {method}: {e}")

    # Plotting execution time and payload fraction; figures are saved to file.
    method_names = [r['method'] for r in results]
    times = [r['time'] for r in results]
    payloads = [r['payload'] for r in results]

    plt.figure(figsize=(10, 5))
    plt.bar(method_names, times)
    plt.title('Execution Time by Solver')
    plt.xlabel('Solver Method')
    plt.ylabel('Time (s)')
    plt.savefig('execution_time.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(method_names, payloads)
    plt.title('Payload Fraction by Solver')
    plt.xlabel('Solver Method')
    plt.ylabel('Payload Fraction')
    plt.savefig('payload_fraction.png')
    plt.close()

    # Print all the results to the console in a neat format.
    print("\nOptimization Results:")
    for result in results:
        print("--------------------------------------------------")
        print(f"Method: {result['method']}")
        if 'time' in result:
            print(f"Execution Time (s): {result['time']:.4f}")
        # Convert NumPy numbers to Python floats for a cleaner printout.
        dv_list = [float(x) for x in result.get('dv', [])]
        ratio_list = [float(x) for x in result.get('ratio', [])]
        print("Optimal Delta-V Allocations:", dv_list)
        print("Optimal Stage Ratios:", ratio_list)
        if 'payload' in result:
            print(f"Overall Payload Fraction: {result['payload']:.4f}")
    print("--------------------------------------------------\n")

    # Generate LaTeX report.
    with open('report.tex', 'w') as f:
        f.write(r"""
\documentclass{article}
\usepackage{graphicx}
\begin{document}
\section{Execution Time}
\includegraphics[width=\textwidth]{execution_time.png}

\section{Payload Fraction}
\includegraphics[width=\textwidth]{payload_fraction.png}

\section{Results}
\begin{tabular}{lccc}
\hline
Method & Time (s) & Payload Fraction \\
\hline
""")
        for result in results:
            f.write(f"{result['method']} & {result['time']:.4f} & {result['payload']:.4f} \\\\\n")
        f.write(r"""
\hline
\end{tabular}
\end{document}
""")
    
    # Plot the ΔV breakdown (including drag and gravity losses) and display it.
    plot_dv_breakdown(results, TOTAL_DELTA_V, GRAVITY_LOSS, DRAG_LOSS, filename="dv_breakdown.png")

if __name__ == '__main__':
    main()
