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
from scipy.optimize import minimize, differential_evolution, approx_fprime
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.core.problem import Problem
import matplotlib.pyplot as plt
import pandas as pd

# Constants for gravity and drag losses (in m/s)
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
            # Switch mode when encountering an empty line.
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

def payload_fraction_objective(dv, G0, ISP, EPSILON, penalty_coeff=1e6):
    """
    Objective function to be minimized.
    Flattens the input so that dv is a 1D array of scalars.
    Returns the negative product of stage ratios.

    Args:
        dv (array-like): Delta-V values for each stage
        G0 (float): Gravitational constant (m/s^2)
        ISP (array-like): Specific impulse values for each stage (s)
        EPSILON (array-like): Mass fraction values for each stage (dimensionless)
        penalty_coeff (float): Penalty coefficient for invalid solutions (default: 1e6)

    Returns:
        float: Negative product of stage ratios, or penalty value if constraints are violated
    """
    dv = np.asarray(dv).flatten()  # Ensure dv is 1D
    
    # Validate ISP values
    if np.any(np.array(ISP) <= 0):
        raise ValueError("ISP values must be positive")
    
    # Check for NaN or infinite values in input
    if np.any(np.isnan(dv)) or np.any(np.isinf(dv)):
        return penalty_coeff
    
    product = 1.0
    for i, dvi in enumerate(dv):
        # Protect against overflow in exponential calculation
        exp_term = -dvi / (G0 * ISP[i])
        if exp_term < -700:  # numpy.exp underflows at ~-709
            return penalty_coeff
            
        # Calculate the stage ratio for this stage
        try:
            f_i = np.exp(exp_term) - EPSILON[i]
        except OverflowError:
            return penalty_coeff
            
        # Return penalty if the stage ratio is not positive
        if f_i <= 0:
            return penalty_coeff
        product *= f_i
    return -product

def objective_with_penalty(dv, G0, ISP, EPSILON, total_delta_v, penalty_coeff=1e6, tol=None):
    """
    A wrapper objective that adds a penalty if the equality constraint
    sum(dv) == total_delta_v is violated.

    Args:
        dv (array-like): Delta-V values for each stage
        G0 (float): Gravitational constant (m/s^2)
        ISP (array-like): Specific impulse values for each stage (s)
        EPSILON (array-like): Mass fraction values for each stage (dimensionless)
        total_delta_v (float): Required total Delta-V
        penalty_coeff (float): Penalty coefficient for constraint violations
        tol (float, optional): Tolerance for constraint satisfaction. If None,
                             defaults to 1e-6 * total_delta_v

    Returns:
        float: Objective value with penalty if constraints are violated
    """
    dv = np.asarray(dv).flatten()
    
    # Set tolerance based on problem scale if not provided
    if tol is None:
        tol = 1e-6 * total_delta_v
    
    # Check for NaN or infinite values
    if np.any(np.isnan(dv)) or np.any(np.isinf(dv)):
        return penalty_coeff
    
    constraint_error = abs(np.sum(dv) - total_delta_v)
    penalty = penalty_coeff * constraint_error if constraint_error > tol else 0.0
    
    # Calculate base objective with same penalty coefficient
    base_objective = payload_fraction_objective(dv, G0, ISP, EPSILON, penalty_coeff)
    
    return base_objective + penalty

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
        tol (float, optional): Tolerance for constraint satisfaction. If None,
                             defaults to 1e-6 * TOTAL_DELTA_V
    
    Returns:
        tuple: (optimal_dv, optimal_stage_ratio, overall_payload)
    """
    n = len(ISP)
    if n != len(EPSILON):
        raise ValueError("ISP and EPSILON lists must have the same length.")
    
    # Validate ISP values
    if np.any(np.array(ISP) <= 0):
        raise ValueError("ISP values must be positive")
    
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
    
    if method.upper() == 'SLSQP':
        constraints = {'type': 'eq', 'fun': lambda dv: np.sum(dv) - TOTAL_DELTA_V}
        res = minimize(payload_fraction_objective, initial_guess, 
                      args=(G0, ISP, EPSILON, penalty_coeff), 
                      method=method, bounds=bounds, constraints=constraints)
        optimal_dv = res.x
    elif method.lower() == 'differential_evolution':
        res = differential_evolution(
            objective_with_penalty, 
            bounds, 
            args=(G0, ISP, EPSILON, TOTAL_DELTA_V, penalty_coeff, tol),
            strategy='best1bin',
            popsize=200,
            tol=1e-6,
            mutation=(0.5, 1),
            recombination=0.7,
            polish=True
        )
        optimal_dv = res.x
    elif method.upper() == 'GA':
        class OptimizationProblem(Problem):
            def __init__(self):
                # Use standard bounds without stage-specific constraints
                super().__init__(n_var=n, n_obj=1, xl=np.zeros(n), xu=max_dv)

            def _evaluate(self, x, out, *args, **kwargs):
                # Evaluate both the objective and how balanced the solution is
                base_objective = objective_with_penalty(x, G0, ISP, EPSILON, TOTAL_DELTA_V, penalty_coeff, tol)
                
                # Add a soft penalty for very unbalanced solutions
                stage_ratios = [np.exp(-dv / (G0 * isp)) - eps for dv, isp, eps in zip(x, ISP, EPSILON)]
                ratio_spread = max(stage_ratios) - min(stage_ratios)
                balance_penalty = ratio_spread * penalty_coeff * 0.1  # Softer penalty for imbalance
                
                out["F"] = base_objective + balance_penalty

        problem = OptimizationProblem()

        # Increase population size for better exploration
        pop_size = 400

        # Generate smarter initial population
        custom_pop = np.zeros((pop_size, n))
        for i in range(pop_size):
            if i < pop_size // 3:
                # First third: roughly equal distribution
                custom_pop[i] = np.full(n, TOTAL_DELTA_V / n) + np.random.normal(0, TOTAL_DELTA_V * 0.05, n)
            elif i < 2 * pop_size // 3:
                # Second third: weighted by ISP (higher ISP gets more Delta-V)
                weights = np.array(ISP) / sum(ISP)
                custom_pop[i] = weights * TOTAL_DELTA_V + np.random.normal(0, TOTAL_DELTA_V * 0.05, n)
            else:
                # Last third: weighted by structural efficiency (lower EPSILON gets more Delta-V)
                weights = (1 - np.array(EPSILON)) / sum(1 - np.array(EPSILON))
                custom_pop[i] = weights * TOTAL_DELTA_V + np.random.normal(0, TOTAL_DELTA_V * 0.05, n)
            
            # Ensure non-negative values and total Delta-V constraint
            custom_pop[i] = np.maximum(custom_pop[i], 0)
            custom_pop[i] = custom_pop[i] * TOTAL_DELTA_V / custom_pop[i].sum()

        algorithm = GA(
            pop_size=pop_size,
            eliminate_duplicates=True,
            mutation=0.15,     # Slightly higher mutation for more exploration
            crossover=0.85,    # High crossover rate for better exploitation
            survival='auto',   # Let the algorithm adapt survival selection
            sampling=custom_pop  # Use our smart initial population
        )

        res_obj = pymoo_minimize(
            problem,
            algorithm,
            termination=('n_gen', 400),  # More generations for better convergence
            seed=42  # Set seed for reproducibility
        )
        res = res_obj.X
        optimal_dv = res
    elif method.lower() == 'newton':
        jacobian = lambda x: approx_fprime(x, lambda dv: objective_with_penalty(dv, G0, ISP, EPSILON, TOTAL_DELTA_V, penalty_coeff, tol), 1e-8)
        res = minimize(objective_with_penalty, initial_guess, args=(G0, ISP, EPSILON, TOTAL_DELTA_V, penalty_coeff, tol), method='Newton-CG', jac=jacobian)
        optimal_dv = res.x
    elif method.lower() == 'dp':
        if n > 3:
            raise ValueError('Dynamic programming approach only supports up to 3 stages')
        if n == 1:
            optimal_dv = np.array([TOTAL_DELTA_V])
        elif n == 2:
            best_obj = np.inf
            best_alloc = None
            steps = 1000
            step_size = TOTAL_DELTA_V / steps
            for i in range(steps+1):
                dv1 = i * step_size
                dv2 = TOTAL_DELTA_V - dv1
                dv = np.array([dv1, dv2])
                obj = objective_with_penalty(dv, G0, ISP, EPSILON, TOTAL_DELTA_V, penalty_coeff, tol)
                if obj < best_obj:
                    best_obj = obj
                    best_alloc = dv
            optimal_dv = best_alloc
        elif n == 3:
            best_obj = np.inf
            best_alloc = None
            resolution = 100  # coarse resolution for 3 stages
            step_size = TOTAL_DELTA_V / resolution
            for i in range(resolution+1):
                for j in range(resolution+1 - i):
                    dv1 = i * step_size
                    dv2 = j * step_size
                    dv3 = TOTAL_DELTA_V - dv1 - dv2
                    if dv3 < 0:
                        continue
                    dv = np.array([dv1, dv2, dv3])
                    obj = objective_with_penalty(dv, G0, ISP, EPSILON, TOTAL_DELTA_V, penalty_coeff, tol)
                    if obj < best_obj:
                        best_obj = obj
                        best_alloc = dv
            optimal_dv = best_alloc
    else:
        raise ValueError(f"Unsupported optimization method: {method}")
 
    stage_ratios = [np.exp(-dvi / (G0 * ISP[i])) - EPSILON[i] for i, dvi in enumerate(optimal_dv)]
    overall_payload = np.prod(stage_ratios)
    
    return optimal_dv, stage_ratios, overall_payload

def plot_dv_breakdown(results, total_delta_v, gravity_loss, drag_loss, filename="dv_breakdown.png"):
    """
    Plot the engine-provided ΔV breakdown per solver, including drag and gravity losses.
    Labels each bar with the stage ratio and total delta-V.
    """
    solver_names = [res["Method"] for res in results]
    indices = np.arange(len(solver_names))
    bar_width = 0.4

    required_engine_dv = total_delta_v + gravity_loss + drag_loss
    colors = ['dodgerblue', 'orange', 'green', 'purple', 'cyan', 'magenta']

    plt.figure(figsize=(12, 6))

    for i, res in enumerate(results):
        sol = res["dv"]      # Allocated ΔV per stage.
        ratios = res["ratio"]  # Stage ratios
        cumulative = 0

        for j, (dv, ratio) in enumerate(zip(sol, ratios)):
            plt.bar(i, dv, bar_width, bottom=cumulative, color=colors[j % len(colors)], 
                    label=f'Stage {j+1}' if i == 0 else "")
            cumulative += dv
            plt.text(i, cumulative - dv / 2, f"{dv:.1f} m/s\n({ratio:.2f})", 
                     ha='center', va='center', fontsize=9, color='white', fontweight='bold')

        plt.text(i, cumulative + 50, f"Total: {cumulative:.0f} m/s", 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.axhline(required_engine_dv, color='red', linestyle='--', linewidth=2, label='Required Engine ΔV')
    plt.xticks(indices, solver_names)
    plt.ylabel("ΔV (m/s)")
    plt.title("ΔV Solution per Solver")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def generate_report(parameters, stages, results):
    """
    Generates a LaTeX report with tables for optimization results, stage ratios, and input data.
    Saves the report as report.tex and also generates CSV files.
    """
    # Create DataFrames
    input_df = pd.DataFrame(list(parameters.items()), columns=["Parameter", "Value"])
    stages_df = pd.DataFrame(stages)
    results_df = pd.DataFrame(results)
    
    # Convert numpy types to Python native types and format floats
    def format_value(x):
        if isinstance(x, (np.floating, np.integer)):
            return float(x)
        elif isinstance(x, (list, np.ndarray)):
            # Format array elements with proper precision
            if isinstance(x, np.ndarray):
                x = x.tolist()
            return ', '.join(f'{val:.2f}' for val in x)
        elif isinstance(x, str) and 'np.float64' in x:
            # Clean up numpy float64 string representations
            return x.replace('np.float64(', '').replace(')', '')
        elif isinstance(x, str) and '[' in x and ']' in x:
            # Handle string representations of arrays
            try:
                # Extract numbers from the string and format them
                nums = [float(n) for n in x.strip('[]').split()]
                return ', '.join(f'{val:.2f}' for val in nums)
            except:
                return x
        return x
    
    results_df = results_df.applymap(format_value)
    stages_df = stages_df.applymap(format_value)
    input_df = input_df.applymap(format_value)
    
    # Save CSV files
    input_df.to_csv("input_variables.csv", index=False)
    stages_df.to_csv("stage_data.csv", index=False)
    results_df.to_csv("optimization_results.csv", index=False)
    
    # Print tables to console
    print("\n--- Input Variables ---")
    print(input_df.to_string(index=False))
    print("\n--- Stage Data ---")
    print(stages_df.to_string(index=False))
    print("\n--- Optimization Results ---")
    print(results_df.to_string(index=False))
    print("\nCSV files generated: input_variables.csv, stage_data.csv, optimization_results.csv")

    # Generate LaTeX report
    latex_doc = [
        "\\documentclass{article}",
        "\\usepackage{booktabs}",
        "\\usepackage{caption}",
        "\\usepackage[margin=1in]{geometry}",
        "\\usepackage{graphicx}",
        "\\usepackage{float}",
        "\\usepackage{array}",
        "\\usepackage{siunitx}",
        "\\sisetup{scientific-notation=true}",
        "\\begin{document}",
        "\\title{Rocket Stage Optimization Report}",
        "\\author{Generated by Payload Optimization Script}",
        "\\maketitle",
        
        "\\section{Problem Definition}",
        "\\subsection{Input Parameters}",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Global Parameters}",
        input_df.to_latex(
            index=False,
            escape=True,
            column_format='lc',
            na_rep='--'
        ).replace('\\toprule', '\\toprule\n\\midrule').replace('\\bottomrule', '\\midrule\n\\bottomrule'),
        "\\end{table}",
        
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Stage Parameters}",
        stages_df.to_latex(
            index=False,
            escape=True,
            float_format=lambda x: '{:.3f}'.format(x),
            column_format='c' * len(stages_df.columns),
            na_rep='--'
        ).replace('\\toprule', '\\toprule\n\\midrule').replace('\\bottomrule', '\\midrule\n\\bottomrule'),
        "\\end{table}",
        
        "\\section{Performance Analysis}",
        "\\subsection{Execution Time}",
        "\\begin{figure}[H]",
        "\\centering",
        "\\includegraphics[width=\\textwidth]{execution_time.png}",
        "\\caption{Execution time comparison between solvers}",
        "\\end{figure}",
        
        "\\subsection{Payload Fraction}",
        "\\begin{figure}[H]",
        "\\centering",
        "\\includegraphics[width=\\textwidth]{payload_fraction.png}",
        "\\caption{Payload fraction comparison between solvers}",
        "\\end{figure}",
        
        "\\subsection{Delta-V Breakdown}",
        "\\begin{figure}[H]",
        "\\centering",
        "\\includegraphics[width=\\textwidth]{dv_breakdown.png}",
        "\\caption{Delta-V breakdown by solver}",
        "\\end{figure}",
        
        "\\section{Optimization Results}",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Optimization Results by Solver}",
        "\\small",
        results_df.to_latex(
            index=False,
            escape=True,
            float_format=lambda x: '{:.6f}'.format(x) if isinstance(x, (float, np.floating)) and abs(x) < 1000 else '{:.2e}'.format(x) if isinstance(x, (float, np.floating)) else str(x),
            column_format='l' + 'c' * (len(results_df.columns) - 1),
            na_rep='--'
        ).replace('\\toprule', '\\toprule\n\\midrule').replace('\\bottomrule', '\\midrule\n\\bottomrule'),
        "\\end{table}",
    ]

    # Add stage ratios if available
    if "Stage Ratios" in results_df.columns:
        latex_doc.extend([
            "\\begin{table}[H]",
            "\\centering",
            "\\caption{Stage Ratios by Solver}",
            results_df[["Solver", "Stage Ratios"]].to_latex(
                index=False,
                escape=True,
                column_format='lc',
                na_rep='--'
            ).replace('\\toprule', '\\toprule\n\\midrule').replace('\\bottomrule', '\\midrule\n\\bottomrule'),
            "\\end{table}"
        ])
    else:
        latex_doc.append("Note: Stage ratios not available in results.")

    latex_doc.append("\\end{document}")

    # Write LaTeX document to file
    with open("report.tex", "w") as f:
        f.write("\n".join(latex_doc))

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

    methods = ['SLSQP', 'differential_evolution', 'GA', 'newton', 'dp']
    optimization_results = []

    # Run optimization for each method.
    for method in methods:
        start_time = time.time()
        try:
            optimal_dv, optimal_stage_ratio, overall_payload = optimize_payload_allocation(
                TOTAL_DELTA_V, ISP, EPSILON, G0, method=method
            )
            execution_time = time.time() - start_time
            optimization_results.append({
                "Method": method,
                "Time (s)": execution_time,
                "Payload Fraction": overall_payload,
                "dv": optimal_dv,
                "ratio": optimal_stage_ratio
            })
        except Exception as e:
            print(f"Error during optimization with {method}: {e}")

    # Generate plots for execution time and payload fraction.
    method_names = [r["Method"] for r in optimization_results]
    times = [r["Time (s)"] for r in optimization_results]
    payloads = [r["Payload Fraction"] for r in optimization_results]

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

    # Plot the ΔV breakdown (including drag and gravity losses).
    plot_dv_breakdown(optimization_results, TOTAL_DELTA_V, GRAVITY_LOSS, DRAG_LOSS, filename="dv_breakdown.png")

    # Print results to console.
    print("\nOptimization Results:")
    for result in optimization_results:
        print("--------------------------------------------------")
        print(f"Method: {result['Method']}")
        print(f"Execution Time (s): {result['Time (s)']:.4f}")
        dv_list = [float(x) for x in result.get('dv', [])]
        ratio_list = [float(x) for x in result.get('ratio', [])]
        print("Optimal Delta-V Allocations:", dv_list)
        print("Optimal Stage Ratios:", ratio_list)
        print(f"Overall Payload Fraction: {result['Payload Fraction']:.4f}")
    print("--------------------------------------------------\n")

    # Generate LaTeX report.
    with open('report.tex', 'w', encoding='utf-8') as f:
        f.write(r"""\documentclass{article}
\usepackage{graphicx}
\begin{document}
\section{Execution Time}
\includegraphics[width=\textwidth]{execution_time.png}

\section{Payload Fraction}
\includegraphics[width=\textwidth]{payload_fraction.png}
                
\section{ΔV Breakdown}
\includegraphics[width=\textwidth]{dv_breakdown.png}

\section{Results}
\begin{tabular}{lccc}
\hline
Method & Time (s) & Payload Fraction \\
\hline
""")
        for result in optimization_results:
            f.write(f"{result['Method']} & {result['Time (s)']:.4f} & {result['Payload Fraction']:.4f} \\\\\n")
        f.write(r"""\hline
\end{tabular}
\end{document}
""")
    
    # Generate additional report (CSV files and DataFrame display).
    generate_report(parameters, stages, optimization_results)

if __name__ == '__main__':
    main()
