#!/usr/bin/env python3
"""Main script for rocket stage optimization."""
import sys
import time
import numpy as np

from src.utils.config import CONFIG, logger, OUTPUT_DIR
from src.utils.data import load_input_data, calculate_mass_ratios, calculate_payload_fraction
from src.optimization.solvers import (
    solve_with_slsqp,
    solve_with_basin_hopping,
    solve_with_differential_evolution,
    solve_with_ga,
    solve_with_adaptive_ga
)
from src.visualization.plots import plot_results
from src.reporting.latex import generate_latex_report


def main():
    """Main optimization routine."""
    try:
        if len(sys.argv) != 2:
            print(f"Usage: {sys.argv[0]} input_data.json")
            sys.exit(1)
        
        # Load input data
        input_file = sys.argv[1]
        parameters, stages = load_input_data(input_file)
        
        # Extract global parameters
        global TOTAL_DELTA_V
        TOTAL_DELTA_V = float(parameters['TOTAL_DELTA_V'])
        G0 = float(parameters.get('G0', 9.81))
        
        # Extract stage parameters
        ISP = [float(stage['ISP']) for stage in stages]
        EPSILON = [float(stage['EPSILON']) for stage in stages]
        n_stages = len(stages)
        
        # Initial setup for optimization
        initial_guess = np.full(n_stages, TOTAL_DELTA_V / n_stages)
        max_dv = TOTAL_DELTA_V * 0.9
        bounds = np.array([(0, max_dv) for _ in range(n_stages)])
        
        # Run optimizations with all methods
        methods = ['SLSQP', 'BASIN-HOPPING', 'GA', 'ADAPTIVE-GA', 'DE']
        results = []
        
        for method in methods:
            try:
                start_time = time.time()
                
                if method == 'GA':
                    optimal_solution = solve_with_ga(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, CONFIG)
                elif method == 'ADAPTIVE-GA':
                    optimal_solution = solve_with_adaptive_ga(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, CONFIG)
                elif method == 'DE':
                    optimal_solution = solve_with_differential_evolution(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, CONFIG)
                else:
                    result = solve_with_slsqp(initial_guess, bounds, G0, ISP, EPSILON, TOTAL_DELTA_V, CONFIG)
                    results.append(result)
                    continue
                
                # For GA and DE methods, format results consistently with other methods
                execution_time = time.time() - start_time
                optimal_solution = np.asarray(optimal_solution).flatten()  # Ensure 1D array
                mass_ratios = calculate_mass_ratios(optimal_solution, ISP, EPSILON, G0)
                payload_fraction = calculate_payload_fraction(mass_ratios)
                
                result = {
                    'method': method,
                    'time': execution_time,
                    'dv': [float(x) for x in optimal_solution],
                    'stage_ratios': [float(x) for x in mass_ratios],
                    'payload_fraction': float(payload_fraction),
                    'error': float(abs(np.sum(optimal_solution) - TOTAL_DELTA_V))
                }
                results.append(result)
                logger.info(f"Successfully completed {method} optimization")
            except Exception as e:
                logger.error(f"Optimization with {method} failed: {e}")
                continue
        
        # Generate plots and report
        if results:
            plot_results(results)
            generate_latex_report(results)
        
    except Exception as e:
        logger.error(f"Error in main routine: {e}")
        raise


if __name__ == '__main__':
    main()
