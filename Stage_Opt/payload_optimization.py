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
import time  # <-- Added missing import for time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.core.problem import Problem
import matplotlib.pyplot as plt

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
    
    We first flatten the input to ensure that dv is a 1D array of scalars.
    """
    dv = np.asarray(dv).flatten()  # Ensure dv is 1D
    product = 1.0
    for i, dvi in enumerate(dv):
        # Calculate the stage ratio for this stage
        f_i = np.exp(-dvi / (G0 * ISP[i])) - EPSILON[i]
        # If f_i is not positive, we return a high penalty value.
        if f_i <= 0:
            return 1e6  # Penalty for an infeasible stage allocation
        product *= f_i
    return -product

def optimize_payload_allocation(TOTAL_DELTA_V, ISP, EPSILON, G0=9.81, method='SLSQP'):
    """Optimize the allocation of TOTAL_DELTA_V among the rocket's stages."""
    n = len(ISP)
    if n != len(EPSILON):
        raise ValueError("ISP and EPSILON lists must have the same length.")
    
    max_dv = np.array([-G0 * isp * np.log(eps) for isp, eps in zip(ISP, EPSILON)])
    if TOTAL_DELTA_V > np.sum(max_dv):
        raise ValueError("TOTAL_DELTA_V exceeds the maximum possible with the given stage constraints.")
    
    if n == 1:
        dv = TOTAL_DELTA_V
        f = np.exp(-dv / (G0 * ISP[0])) - EPSILON[0]
        return [dv], [f], f
    
    initial_guess = np.full(n, TOTAL_DELTA_V / n)
    bounds = [(0, max_dv_i) for max_dv_i in max_dv]
    constraints = {'type': 'eq', 'fun': lambda dv: np.sum(dv) - TOTAL_DELTA_V}

    if method == 'SLSQP':
        res = minimize(payload_fraction_objective, initial_guess, args=(G0, ISP, EPSILON), 
                       method=method, bounds=bounds, constraints=constraints)
    elif method == 'differential_evolution':
        res = differential_evolution(payload_fraction_objective, bounds, args=(G0, ISP, EPSILON))
    elif method == 'GA':
        class OptimizationProblem(Problem):
            def __init__(self):
                super().__init__(n_var=n, n_obj=1, xl=np.zeros(n), xu=max_dv)
            def _evaluate(self, x, out, *args, **kwargs):
                # Ensure x is 2D. If it isn't, reshape it.
                if x.ndim == 1:
                    x = x.reshape(1, -1)
                # Flatten each candidate solution to avoid ambiguous truth values.
                F = np.array([payload_fraction_objective(np.asarray(xi).flatten(), G0, ISP, EPSILON) for xi in x])
                out["F"] = F.reshape(-1, 1)
        
        problem = OptimizationProblem()
        algorithm = GA(pop_size=50)
        res_obj = pymoo_minimize(problem, algorithm, termination=('n_gen', 50))
        res = res_obj.X  # GA returns the best solution directly

    if method in ['SLSQP', 'differential_evolution']:
        if not res.success:
            raise RuntimeError(f"Optimization failed: {res.message}")
        optimal_dv = res.x
    else:
        optimal_dv = res

    optimal_stage_ratio = [np.exp(-dvi / (G0 * ISP[i])) - EPSILON[i] for i, dvi in enumerate(optimal_dv)]
    overall_payload = np.prod(optimal_stage_ratio)
    
    return optimal_dv, optimal_stage_ratio, overall_payload

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

    # Plotting results: write figures to file and do not display them.
    method_names = [r['method'] for r in results]
    times = [r['time'] for r in results]
    payloads = [r['payload'] for r in results]

    plt.figure(figsize=(10, 5))
    plt.bar(method_names, times)
    plt.title('Execution Time by Solver')
    plt.xlabel('Solver Method')
    plt.ylabel('Time (s)')
    plt.savefig('execution_time.png')
    plt.close()  # Close the figure

    plt.figure(figsize=(10, 5))
    plt.bar(method_names, payloads)
    plt.title('Payload Fraction by Solver')
    plt.xlabel('Solver Method')
    plt.ylabel('Payload Fraction')
    plt.savefig('payload_fraction.png')
    plt.close()  # Close the figure

    # Generate LaTeX report
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

if __name__ == '__main__':
    main()
